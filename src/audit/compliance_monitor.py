"""
Real-time compliance monitoring and violation detection system.

This module provides comprehensive compliance monitoring with configurable rules,
real-time violation detection, automated alerting, and regulatory standard support
for enterprise-grade compliance management.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Callable, Tuple
from datetime import datetime, timedelta, UTC
import asyncio
import logging
import time

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.audit_framework import (
    AuditEvent, AuditEventType, ComplianceRule, ComplianceStandard,
    RiskLevel, AuditError
)


logger = logging.getLogger(__name__)


class ComplianceMonitor:
    """Real-time compliance monitoring with violation detection and alerting."""
    
    def __init__(self):
        self.compliance_rules: Dict[str, ComplianceRule] = {}
        self.violation_callbacks: List[Callable] = []
        self.monitoring_enabled = True
        self.violation_cache: Dict[str, datetime] = {}  # Prevent duplicate alerts
        self.cache_expiry = timedelta(minutes=5)
        
        # Statistics tracking
        self.stats = {
            'events_monitored': 0,
            'violations_detected': 0,
            'rules_evaluated': 0,
            'false_positives': 0,
            'start_time': datetime.now(UTC)
        }
    
    @require(lambda self, rule: isinstance(rule, ComplianceRule))
    async def register_compliance_rule(self, rule: ComplianceRule) -> Either[AuditError, None]:
        """Register new compliance monitoring rule with validation."""
        try:
            # Validate rule configuration
            validation_result = self._validate_rule(rule)
            if validation_result.is_left():
                return validation_result
            
            self.compliance_rules[rule.rule_id] = rule
            logger.info(f"Registered compliance rule: {rule.name} ({rule.standard.value})")
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AuditError.rule_registration_failed(str(e)))
    
    async def unregister_rule(self, rule_id: str) -> bool:
        """Unregister compliance rule."""
        if rule_id in self.compliance_rules:
            rule = self.compliance_rules.pop(rule_id)
            logger.info(f"Unregistered compliance rule: {rule.name}")
            return True
        return False
    
    @require(lambda self, event: isinstance(event, AuditEvent))
    async def monitor_event(self, event: AuditEvent) -> List[ComplianceRule]:
        """Monitor event for compliance violations with performance optimization."""
        violations = []
        
        if not self.monitoring_enabled:
            return violations
        
        try:
            self.stats['events_monitored'] += 1
            
            # Check each enabled rule
            for rule in self.compliance_rules.values():
                if not rule.enabled:
                    continue
                
                self.stats['rules_evaluated'] += 1
                
                # Evaluate rule against event
                try:
                    if rule.evaluate(event):
                        # Check for duplicate violation (anti-spam)
                        cache_key = f"{rule.rule_id}:{event.user_id}:{event.action}"
                        if not self._is_duplicate_violation(cache_key):
                            violations.append(rule)
                            self.stats['violations_detected'] += 1
                            
                            # Cache violation to prevent duplicates
                            self.violation_cache[cache_key] = datetime.now(UTC)
                            
                            # Trigger violation callbacks
                            await self._trigger_violation_callbacks(event, rule)
                            
                            logger.warning(f"Compliance violation detected: {rule.name} - {event.action}")
                    
                except Exception as e:
                    logger.error(f"Error evaluating rule {rule.rule_id}: {e}")
                    continue
            
            # Clean up old cache entries
            await self._cleanup_violation_cache()
            
        except Exception as e:
            logger.error(f"Error monitoring event: {e}")
        
        return violations
    
    async def evaluate_batch(self, events: List[AuditEvent]) -> Dict[str, List[ComplianceRule]]:
        """Evaluate multiple events for compliance violations efficiently."""
        results = {}
        
        for event in events:
            violations = await self.monitor_event(event)
            if violations:
                results[event.event_id] = violations
        
        return results
    
    def enable_monitoring(self):
        """Enable compliance monitoring."""
        self.monitoring_enabled = True
        logger.info("Compliance monitoring enabled")
    
    def disable_monitoring(self):
        """Disable compliance monitoring."""
        self.monitoring_enabled = False
        logger.info("Compliance monitoring disabled")
    
    def add_violation_callback(self, callback: Callable):
        """Add callback to be called when violations are detected."""
        self.violation_callbacks.append(callback)
    
    def remove_violation_callback(self, callback: Callable):
        """Remove violation callback."""
        if callback in self.violation_callbacks:
            self.violation_callbacks.remove(callback)
    
    def get_rules_by_standard(self, standard: ComplianceStandard) -> List[ComplianceRule]:
        """Get all rules for specific compliance standard."""
        return [rule for rule in self.compliance_rules.values() 
                if rule.standard == standard]
    
    def get_active_rules(self) -> List[ComplianceRule]:
        """Get all active (enabled) compliance rules."""
        return [rule for rule in self.compliance_rules.values() if rule.enabled]
    
    def load_standard_rules(self, standard: ComplianceStandard):
        """Load pre-defined compliance rules for specific standard."""
        standard_rules = self._get_standard_rules(standard)
        
        for rule in standard_rules:
            # Use asyncio to register rules
            asyncio.create_task(self.register_compliance_rule(rule))
        
        logger.info(f"Loaded {len(standard_rules)} standard rules for {standard.value}")
    
    def _get_standard_rules(self, standard: ComplianceStandard) -> List[ComplianceRule]:
        """Get pre-defined rules for compliance standards."""
        rules_map = {
            ComplianceStandard.HIPAA: self._get_hipaa_rules(),
            ComplianceStandard.GDPR: self._get_gdpr_rules(),
            ComplianceStandard.PCI_DSS: self._get_pci_rules(),
            ComplianceStandard.SOC2: self._get_soc2_rules(),
            ComplianceStandard.ISO_27001: self._get_iso27001_rules(),
        }
        
        return rules_map.get(standard, [])
    
    def _get_hipaa_rules(self) -> List[ComplianceRule]:
        """Get HIPAA compliance rules."""
        return [
            ComplianceRule(
                rule_id="hipaa_001",
                name="Unauthorized PHI Access",
                description="Detect unauthorized access to protected health information",
                standard=ComplianceStandard.HIPAA,
                severity=RiskLevel.HIGH,
                condition="sensitive_data AND NOT authorized",
                action="alert_security_team",
                tags={'phi_access', 'unauthorized'}
            ),
            ComplianceRule(
                rule_id="hipaa_002",
                name="Failed PHI System Authentication",
                description="Monitor failed login attempts to PHI systems",
                standard=ComplianceStandard.HIPAA,
                severity=RiskLevel.MEDIUM,
                condition="failed_login AND medical_system",
                action="log_security_event",
                tags={'authentication', 'phi_access'}
            ),
            ComplianceRule(
                rule_id="hipaa_003",
                name="PHI Data Export Without Authorization",
                description="Detect exports of protected health information",
                standard=ComplianceStandard.HIPAA,
                severity=RiskLevel.CRITICAL,
                condition="tag:phi_access AND data_export",
                action="block_and_alert",
                tags={'phi_access', 'data_export'}
            ),
            ComplianceRule(
                rule_id="hipaa_004",
                name="Excessive PHI Access",
                description="Detect unusual patterns of PHI access",
                standard=ComplianceStandard.HIPAA,
                severity=RiskLevel.HIGH,
                condition="high_risk AND tag:phi_access",
                action="security_review",
                tags={'phi_access', 'unusual_pattern'}
            )
        ]
    
    def _get_gdpr_rules(self) -> List[ComplianceRule]:
        """Get GDPR compliance rules."""
        return [
            ComplianceRule(
                rule_id="gdpr_001",
                name="Personal Data Processing Without Consent",
                description="Detect processing of personal data without proper consent",
                standard=ComplianceStandard.GDPR,
                severity=RiskLevel.HIGH,
                condition="tag:personal_data AND NOT consent_verified",
                action="block_processing",
                tags={'personal_data', 'consent'}
            ),
            ComplianceRule(
                rule_id="gdpr_002",
                name="Data Retention Violation",
                description="Detect retention of personal data beyond allowed period",
                standard=ComplianceStandard.GDPR,
                severity=RiskLevel.MEDIUM,
                condition="tag:personal_data AND retention_expired",
                action="schedule_deletion",
                tags={'personal_data', 'retention'}
            ),
            ComplianceRule(
                rule_id="gdpr_003",
                name="Cross-Border Data Transfer",
                description="Monitor international personal data transfers",
                standard=ComplianceStandard.GDPR,
                severity=RiskLevel.HIGH,
                condition="tag:personal_data AND international_transfer",
                action="validate_adequacy",
                tags={'personal_data', 'cross_border'}
            )
        ]
    
    def _get_pci_rules(self) -> List[ComplianceRule]:
        """Get PCI-DSS compliance rules."""
        return [
            ComplianceRule(
                rule_id="pci_001",
                name="Payment Data Access",
                description="Monitor access to payment card information",
                standard=ComplianceStandard.PCI_DSS,
                severity=RiskLevel.HIGH,
                condition="tag:payment_data",
                action="log_and_monitor",
                tags={'payment_data', 'card_processing'}
            ),
            ComplianceRule(
                rule_id="pci_002",
                name="Unencrypted Payment Data",
                description="Detect unencrypted payment card data",
                standard=ComplianceStandard.PCI_DSS,
                severity=RiskLevel.CRITICAL,
                condition="tag:payment_data AND NOT encrypted",
                action="immediate_encryption",
                tags={'payment_data', 'encryption'}
            )
        ]
    
    def _get_soc2_rules(self) -> List[ComplianceRule]:
        """Get SOC2 compliance rules."""
        return [
            ComplianceRule(
                rule_id="soc2_001",
                name="Privilege Escalation",
                description="Monitor privilege escalation activities",
                standard=ComplianceStandard.SOC2,
                severity=RiskLevel.HIGH,
                condition="privilege_escalation",
                action="security_review",
                tags={'security', 'privilege_escalation'}
            ),
            ComplianceRule(
                rule_id="soc2_002",
                name="System Configuration Changes",
                description="Monitor unauthorized system configuration changes",
                standard=ComplianceStandard.SOC2,
                severity=RiskLevel.MEDIUM,
                condition="config_change AND NOT authorized",
                action="change_review",
                tags={'configuration', 'unauthorized'}
            )
        ]
    
    def _get_iso27001_rules(self) -> List[ComplianceRule]:
        """Get ISO 27001 compliance rules."""
        return [
            ComplianceRule(
                rule_id="iso27001_001",
                name="Information Security Incident",
                description="Detect information security incidents",
                standard=ComplianceStandard.ISO_27001,
                severity=RiskLevel.HIGH,
                condition="security_violation",
                action="incident_response",
                tags={'information_security', 'incident'}
            ),
            ComplianceRule(
                rule_id="iso27001_002",
                name="Access Control Violation",
                description="Monitor access control violations",
                standard=ComplianceStandard.ISO_27001,
                severity=RiskLevel.MEDIUM,
                condition="unauthorized_access",
                action="access_review",
                tags={'access_control', 'violation'}
            )
        ]
    
    def _validate_rule(self, rule: ComplianceRule) -> Either[AuditError, None]:
        """Validate compliance rule configuration."""
        try:
            # Check for duplicate rule ID
            if rule.rule_id in self.compliance_rules:
                return Either.left(AuditError.rule_registration_failed(
                    f"Rule ID already exists: {rule.rule_id}"
                ))
            
            # Validate rule condition syntax
            if not rule.condition or len(rule.condition.strip()) == 0:
                return Either.left(AuditError.rule_registration_failed(
                    "Rule condition cannot be empty"
                ))
            
            # Validate rule name
            if not rule.name or len(rule.name.strip()) == 0:
                return Either.left(AuditError.rule_registration_failed(
                    "Rule name cannot be empty"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(AuditError.rule_registration_failed(str(e)))
    
    def _is_duplicate_violation(self, cache_key: str) -> bool:
        """Check if violation is a duplicate within cache expiry period."""
        if cache_key not in self.violation_cache:
            return False
        
        cached_time = self.violation_cache[cache_key]
        return datetime.now(UTC) - cached_time < self.cache_expiry
    
    async def _trigger_violation_callbacks(self, event: AuditEvent, rule: ComplianceRule):
        """Trigger violation callbacks asynchronously."""
        for callback in self.violation_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(event, rule)
                else:
                    callback(event, rule)
            except Exception as e:
                logger.error(f"Error in violation callback: {e}")
    
    async def _cleanup_violation_cache(self):
        """Clean up expired violation cache entries."""
        current_time = datetime.now(UTC)
        expired_keys = []
        
        for cache_key, cached_time in self.violation_cache.items():
            if current_time - cached_time >= self.cache_expiry:
                expired_keys.append(cache_key)
        
        for key in expired_keys:
            del self.violation_cache[key]
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get comprehensive monitoring statistics."""
        uptime = datetime.now(UTC) - self.stats['start_time']
        
        return {
            'monitoring_enabled': self.monitoring_enabled,
            'events_monitored': self.stats['events_monitored'],
            'violations_detected': self.stats['violations_detected'],
            'rules_evaluated': self.stats['rules_evaluated'],
            'false_positives': self.stats['false_positives'],
            'active_rules': len(self.get_active_rules()),
            'total_rules': len(self.compliance_rules),
            'violation_cache_size': len(self.violation_cache),
            'uptime_seconds': uptime.total_seconds(),
            'violation_rate': (self.stats['violations_detected'] / 
                             max(self.stats['events_monitored'], 1)) * 100,
            'rules_by_standard': self._get_rules_by_standard_stats()
        }
    
    def _get_rules_by_standard_stats(self) -> Dict[str, int]:
        """Get rule count statistics by compliance standard."""
        stats = {}
        for rule in self.compliance_rules.values():
            standard = rule.standard.value
            stats[standard] = stats.get(standard, 0) + 1
        return stats
    
    async def generate_compliance_summary(self, 
                                        period_start: datetime, 
                                        period_end: datetime) -> Dict[str, Any]:
        """Generate compliance monitoring summary for period."""
        try:
            summary = {
                'period_start': period_start,
                'period_end': period_end,
                'monitoring_enabled': self.monitoring_enabled,
                'active_rules': len(self.get_active_rules()),
                'standards_monitored': list(set(rule.standard.value 
                                              for rule in self.compliance_rules.values())),
                'violation_summary': {
                    'total_violations': self.stats['violations_detected'],
                    'critical_violations': 0,
                    'high_risk_violations': 0,
                    'medium_risk_violations': 0,
                    'low_risk_violations': 0
                },
                'top_violated_rules': [],
                'compliance_score': 0.0
            }
            
            # Calculate compliance score (simplified)
            if self.stats['events_monitored'] > 0:
                compliance_score = max(0.0, 
                    (1.0 - (self.stats['violations_detected'] / self.stats['events_monitored'])) * 100
                )
                summary['compliance_score'] = compliance_score
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating compliance summary: {e}")
            return {}


class ViolationNotifier:
    """Notification system for compliance violations."""
    
    def __init__(self):
        self.notification_channels: Dict[str, Callable] = {}
        self.severity_thresholds = {
            RiskLevel.LOW: 'info',
            RiskLevel.MEDIUM: 'warning',
            RiskLevel.HIGH: 'error',
            RiskLevel.CRITICAL: 'critical'
        }
    
    def register_channel(self, name: str, handler: Callable):
        """Register notification channel."""
        self.notification_channels[name] = handler
    
    async def notify_violation(self, event: AuditEvent, rule: ComplianceRule):
        """Send violation notification through all channels."""
        notification_level = self.severity_thresholds.get(rule.severity, 'info')
        
        message = {
            'type': 'compliance_violation',
            'level': notification_level,
            'rule_id': rule.rule_id,
            'rule_name': rule.name,
            'standard': rule.standard.value,
            'severity': rule.severity.value,
            'event_id': event.event_id,
            'user_id': event.user_id,
            'action': event.action,
            'timestamp': event.timestamp.isoformat(),
            'details': rule.description
        }
        
        for channel_name, handler in self.notification_channels.items():
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(message)
                else:
                    handler(message)
            except Exception as e:
                logger.error(f"Error sending notification via {channel_name}: {e}")
    
    async def send_email_notification(self, message: Dict[str, Any]):
        """Send email notification (placeholder)."""
        # In production, integrate with email service
        logger.info(f"Email notification: {message['rule_name']} violation")
    
    async def send_slack_notification(self, message: Dict[str, Any]):
        """Send Slack notification (placeholder)."""
        # In production, integrate with Slack API
        logger.info(f"Slack notification: {message['rule_name']} violation")
    
    async def log_notification(self, message: Dict[str, Any]):
        """Log notification to system logs."""
        level = message['level']
        text = f"Compliance violation: {message['rule_name']} - {message['details']}"
        
        if level == 'critical':
            logger.critical(text)
        elif level == 'error':
            logger.error(text)
        elif level == 'warning':
            logger.warning(text)
        else:
            logger.info(text)