"""
Comprehensive audit system management and coordination.

This module provides the main audit system manager that coordinates all audit
components including event logging, compliance monitoring, report generation,
and integration with the broader automation ecosystem.
"""

from __future__ import annotations
from typing import Dict, Any, List, Optional, Set, Callable
from datetime import datetime, timedelta, UTC
import asyncio
import logging
import time

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.audit_framework import (
    AuditEvent, AuditEventType, AuditError, RiskLevel, 
    ComplianceStandard, AuditConfiguration, SecurityLimits
)
from .event_logger import EventLogger
from .compliance_monitor import ComplianceMonitor, ViolationNotifier
from .report_generator import ReportGenerator


logger = logging.getLogger(__name__)


class AuditSystemManager:
    """Comprehensive audit system management and coordination."""
    
    def __init__(self, config: Optional[AuditConfiguration] = None):
        self.config = config or AuditConfiguration()
        
        # Initialize core components
        self.event_logger = EventLogger(self.config)
        self.compliance_monitor = ComplianceMonitor()
        self.report_generator = ReportGenerator(self.event_logger, self.compliance_monitor)
        self.violation_notifier = ViolationNotifier()
        
        # System state
        self.initialized = False
        self.tool_audit_hooks: Dict[str, Callable] = {}
        self.background_tasks: List[asyncio.Task] = []
        
        # Performance tracking
        self.performance_metrics = {
            'audit_latency': [],
            'events_per_second': 0,
            'compliance_checks_per_second': 0,
            'last_performance_check': time.time()
        }
        
        # Setup violation notification
        self.violation_notifier.register_channel('log', self.violation_notifier.log_notification)
        self.compliance_monitor.add_violation_callback(self.violation_notifier.notify_violation)
    
    @require(lambda self, compliance_standards: isinstance(compliance_standards, list))
    async def initialize(self, compliance_standards: Optional[List[ComplianceStandard]] = None) -> Either[AuditError, None]:
        """Initialize audit system with compliance standards and background services."""
        try:
            if self.initialized:
                return Either.right(None)
            
            logger.info("Initializing audit system...")
            
            # Load standard compliance rules
            standards_to_load = compliance_standards or [ComplianceStandard.GENERAL]
            for standard in standards_to_load:
                self.compliance_monitor.load_standard_rules(standard)
                logger.info(f"Loaded compliance rules for {standard.value}")
            
            # Start background services
            await self._start_background_services()
            
            # Register with existing tools
            await self._register_tool_audit_hooks()
            
            self.initialized = True
            
            logger.info(f"Audit system initialized with {len(standards_to_load)} compliance standards")
            return Either.right(None)
            
        except Exception as e:
            logger.error(f"Failed to initialize audit system: {e}")
            return Either.left(AuditError.initialization_failed(str(e)))
    
    async def shutdown(self):
        """Gracefully shutdown audit system and background services."""
        try:
            logger.info("Shutting down audit system...")
            
            # Cancel background tasks
            for task in self.background_tasks:
                task.cancel()
            
            # Wait for tasks to complete
            if self.background_tasks:
                await asyncio.gather(*self.background_tasks, return_exceptions=True)
            
            # Flush any remaining events
            await self.event_logger.buffer.force_flush()
            
            self.initialized = False
            logger.info("Audit system shutdown complete")
            
        except Exception as e:
            logger.error(f"Error during audit system shutdown: {e}")
    
    @require(lambda self, tool_name: isinstance(tool_name, str) and len(tool_name.strip()) > 0)
    @require(lambda self, user_id: isinstance(user_id, str) and len(user_id.strip()) > 0)
    async def audit_tool_execution(self, 
                                  tool_name: str, 
                                  user_id: str, 
                                  parameters: Dict[str, Any], 
                                  result: Dict[str, Any],
                                  execution_time: float = 0.0,
                                  session_id: Optional[str] = None,
                                  ip_address: Optional[str] = None) -> Either[AuditError, str]:
        """Audit tool execution for comprehensive compliance tracking."""
        start_time = time.time()
        
        try:
            # Determine event type and risk level
            event_type = AuditEventType.TOOL_EXECUTED
            risk_level = self._assess_tool_risk(tool_name, parameters, result)
            
            # Extract compliance tags
            compliance_tags = self._extract_compliance_tags(tool_name, parameters, result)
            
            # Determine result status
            result_status = "success" if result.get('success', False) else "failure"
            if result.get('error'):
                result_status = "error"
            
            # Create detailed audit information
            audit_details = {
                'tool_name': tool_name,
                'parameters': self._sanitize_parameters(parameters),
                'result': self._sanitize_result(result),
                'execution_time_ms': execution_time * 1000,
                'parameter_count': len(parameters),
                'result_size': len(str(result))
            }
            
            # Log the audit event
            audit_result = await self.event_logger.log_event(
                event_type=event_type,
                user_id=user_id,
                action=f"execute_{tool_name}",
                result=result_status,
                session_id=session_id,
                resource_id=tool_name,
                ip_address=ip_address,
                details=audit_details,
                risk_level=risk_level,
                compliance_tags=compliance_tags
            )
            
            if audit_result.is_left():
                return audit_result
            
            event_id = audit_result.get_right()
            
            # Monitor for compliance violations if compliance monitoring is enabled
            if self.compliance_monitor.monitoring_enabled:
                # Create audit event for compliance monitoring
                audit_event = AuditEvent(
                    event_id=event_id,
                    event_type=event_type,
                    timestamp=datetime.now(UTC),
                    user_id=user_id,
                    session_id=session_id,
                    resource_id=tool_name,
                    action=f"execute_{tool_name}",
                    result=result_status,
                    ip_address=ip_address,
                    details=audit_details,
                    risk_level=risk_level,
                    compliance_tags=compliance_tags
                )
                
                # Check for violations
                violations = await self.compliance_monitor.monitor_event(audit_event)
                if violations:
                    logger.warning(f"Compliance violations detected for tool {tool_name}: "
                                 f"{[v.name for v in violations]}")
            
            # Track performance
            audit_latency = time.time() - start_time
            self.performance_metrics['audit_latency'].append(audit_latency)
            
            # Keep only last 1000 latency measurements
            if len(self.performance_metrics['audit_latency']) > 1000:
                self.performance_metrics['audit_latency'] = self.performance_metrics['audit_latency'][-1000:]
            
            return Either.right(event_id)
            
        except Exception as e:
            logger.error(f"Error auditing tool execution for {tool_name}: {e}")
            return Either.left(AuditError.logging_failed(str(e)))
    
    @require(lambda self, event_type: isinstance(event_type, AuditEventType))
    @require(lambda self, user_id: isinstance(user_id, str) and len(user_id.strip()) > 0)
    async def audit_user_action(self,
                               event_type: AuditEventType,
                               user_id: str,
                               action: str,
                               result: str,
                               **kwargs) -> Either[AuditError, str]:
        """Audit general user actions for comprehensive activity tracking."""
        try:
            # Extract additional parameters
            session_id = kwargs.get('session_id')
            resource_id = kwargs.get('resource_id')
            ip_address = kwargs.get('ip_address')
            user_agent = kwargs.get('user_agent')
            details = kwargs.get('details', {})
            risk_level = kwargs.get('risk_level', RiskLevel.LOW)
            compliance_tags = set(kwargs.get('compliance_tags', []))
            
            # Log the audit event
            return await self.event_logger.log_event(
                event_type=event_type,
                user_id=user_id,
                action=action,
                result=result,
                session_id=session_id,
                resource_id=resource_id,
                ip_address=ip_address,
                user_agent=user_agent,
                details=details,
                risk_level=risk_level,
                compliance_tags=compliance_tags
            )
            
        except Exception as e:
            logger.error(f"Error auditing user action: {e}")
            return Either.left(AuditError.logging_failed(str(e)))
    
    async def generate_compliance_report(self,
                                       standard: ComplianceStandard,
                                       period_start: datetime,
                                       period_end: datetime) -> Either[AuditError, Dict[str, Any]]:
        """Generate compliance report and return formatted results."""
        try:
            report_result = await self.report_generator.generate_compliance_report(
                standard, period_start, period_end
            )
            
            if report_result.is_left():
                return report_result
            
            report = report_result.get_right()
            
            # Convert to dictionary format for API response
            report_data = {
                'report_id': report.report_id,
                'standard': report.standard.value,
                'period': {
                    'start': report.period_start.isoformat(),
                    'end': report.period_end.isoformat()
                },
                'summary': {
                    'total_events': report.total_events,
                    'violations_found': report.violations_found,
                    'compliance_percentage': report.compliance_percentage,
                    'risk_score': report.risk_score,
                    'compliance_grade': report.get_compliance_grade(),
                    'risk_category': report.get_risk_category(),
                    'is_compliant': report.is_compliant()
                },
                'findings': report.findings,
                'recommendations': report.recommendations,
                'generated_at': report.generated_at.isoformat()
            }
            
            return Either.right(report_data)
            
        except Exception as e:
            logger.error(f"Error generating compliance report: {e}")
            return Either.left(AuditError.report_generation_failed(str(e)))
    
    async def query_audit_events(self,
                                filters: Dict[str, Any],
                                time_range: Optional[Tuple[datetime, datetime]] = None,
                                limit: int = 100) -> List[Dict[str, Any]]:
        """Query audit events with formatting for API responses."""
        try:
            events = await self.event_logger.query_events(filters, time_range, limit)
            
            # Convert events to dictionary format
            formatted_events = []
            for event in events:
                event_dict = {
                    'event_id': event.event_id,
                    'event_type': event.event_type.value,
                    'timestamp': event.timestamp.isoformat(),
                    'user_id': event.user_id,
                    'session_id': event.session_id,
                    'resource_id': event.resource_id,
                    'action': event.action,
                    'result': event.result,
                    'ip_address': event.ip_address,
                    'user_agent': event.user_agent,
                    'risk_level': event.risk_level.value,
                    'compliance_tags': list(event.compliance_tags),
                    'details': event.details,
                    'integrity_verified': event.verify_integrity()
                }
                formatted_events.append(event_dict)
            
            return formatted_events
            
        except Exception as e:
            logger.error(f"Error querying audit events: {e}")
            return []
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive audit system status."""
        try:
            # Get component statistics
            logger_stats = self.event_logger.get_statistics()
            monitor_stats = self.compliance_monitor.get_statistics()
            
            # Calculate performance metrics
            current_time = time.time()
            time_delta = current_time - self.performance_metrics['last_performance_check']
            
            if time_delta > 0:
                self.performance_metrics['events_per_second'] = (
                    logger_stats['events_logged'] / max(logger_stats['uptime_seconds'], 1)
                )
                self.performance_metrics['compliance_checks_per_second'] = (
                    monitor_stats['rules_evaluated'] / max(monitor_stats['uptime_seconds'], 1)
                )
            
            avg_latency = 0.0
            if self.performance_metrics['audit_latency']:
                avg_latency = sum(self.performance_metrics['audit_latency']) / len(self.performance_metrics['audit_latency'])
            
            return {
                'initialized': self.initialized,
                'configuration': {
                    'audit_level': self.config.audit_level.value,
                    'retention_days': self.config.retention_days,
                    'encryption_enabled': self.config.encrypt_logs,
                    'monitoring_enabled': self.config.enable_real_time_monitoring
                },
                'event_logging': logger_stats,
                'compliance_monitoring': monitor_stats,
                'performance': {
                    'average_audit_latency_ms': avg_latency * 1000,
                    'events_per_second': self.performance_metrics['events_per_second'],
                    'compliance_checks_per_second': self.performance_metrics['compliance_checks_per_second']
                },
                'background_services': {
                    'active_tasks': len([t for t in self.background_tasks if not t.done()]),
                    'total_tasks': len(self.background_tasks)
                }
            }
            
        except Exception as e:
            logger.error(f"Error getting system status: {e}")
            return {'error': str(e)}
    
    def _assess_tool_risk(self, tool_name: str, parameters: Dict[str, Any], result: Dict[str, Any]) -> RiskLevel:
        """Assess risk level of tool execution."""
        # High-risk tools that require elevated monitoring
        high_risk_tools = {
            'km_file_operations', 'km_app_control', 'km_system_control',
            'km_security_manager', 'km_enterprise_sync', 'km_audit_system',
            'km_plugin_ecosystem', 'km_autonomous_agent'
        }
        
        # Medium-risk tools that involve automation changes
        medium_risk_tools = {
            'km_create_macro', 'km_modify_macro', 'km_delete_macro',
            'km_web_automation', 'km_remote_triggers', 'km_add_action',
            'km_control_flow', 'km_create_trigger_advanced'
        }
        
        # Check tool name risk level
        if tool_name in high_risk_tools:
            base_risk = RiskLevel.HIGH
        elif tool_name in medium_risk_tools:
            base_risk = RiskLevel.MEDIUM
        else:
            base_risk = RiskLevel.LOW
        
        # Escalate risk based on operation results
        if result.get('error') or not result.get('success', True):
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH
        
        # Check for sensitive data in parameters
        sensitive_keywords = {'password', 'secret', 'token', 'key', 'credential'}
        param_text = str(parameters).lower()
        if any(keyword in param_text for keyword in sensitive_keywords):
            if base_risk == RiskLevel.LOW:
                base_risk = RiskLevel.MEDIUM
            elif base_risk == RiskLevel.MEDIUM:
                base_risk = RiskLevel.HIGH
        
        return base_risk
    
    def _extract_compliance_tags(self, 
                               tool_name: str, 
                               parameters: Dict[str, Any], 
                               result: Dict[str, Any]) -> Set[str]:
        """Extract compliance-relevant tags from tool execution."""
        tags = set()
        
        # Tool-specific compliance tags
        if 'file_operations' in tool_name:
            tags.add('data_access')
            # Check for potential PHI access
            param_text = str(parameters).lower()
            if any(keyword in param_text for keyword in ['patient', 'medical', 'health', 'phi']):
                tags.add('phi_access')
            if any(keyword in param_text for keyword in ['personal', 'private', 'pii']):
                tags.add('personal_data')
        
        if 'email' in tool_name or 'communication' in tool_name:
            tags.add('communication')
            tags.add('personal_data')
        
        if 'payment' in tool_name or 'financial' in tool_name:
            tags.add('payment_data')
            tags.add('financial')
        
        if 'web' in tool_name or 'api' in tool_name:
            tags.add('external_communication')
            tags.add('data_processing')
        
        if 'plugin' in tool_name or 'enterprise' in tool_name:
            tags.add('system_modification')
            tags.add('security')
        
        # Result-based tags
        if result.get('data_exported'):
            tags.add('data_export')
        
        if result.get('configuration_changed'):
            tags.add('configuration_change')
        
        if result.get('error'):
            tags.add('system_error')
        
        return tags
    
    def _sanitize_parameters(self, parameters: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize parameters for audit logging by masking sensitive data."""
        sensitive_keys = {'password', 'secret', 'token', 'key', 'credential', 'auth'}
        sanitized = {}
        
        for key, value in parameters.items():
            if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                sanitized[key] = '[REDACTED]'
            elif isinstance(value, str) and len(value) > 100:
                # Truncate very long strings
                sanitized[key] = value[:100] + '...[TRUNCATED]'
            else:
                sanitized[key] = value
        
        return sanitized
    
    def _sanitize_result(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Sanitize result data for audit logging."""
        sanitized = {}
        
        for key, value in result.items():
            if key in ['data', 'content', 'response'] and isinstance(value, str) and len(value) > 500:
                # Truncate large result data
                sanitized[key] = value[:500] + '...[TRUNCATED]'
            else:
                sanitized[key] = value
        
        return sanitized
    
    async def _start_background_services(self):
        """Start background services for audit system maintenance."""
        try:
            # Start performance monitoring task
            perf_task = asyncio.create_task(self._performance_monitoring_loop())
            self.background_tasks.append(perf_task)
            
            # Start cache cleanup task
            cache_task = asyncio.create_task(self._cache_cleanup_loop())
            self.background_tasks.append(cache_task)
            
            logger.info("Started audit system background services")
            
        except Exception as e:
            logger.error(f"Error starting background services: {e}")
    
    async def _performance_monitoring_loop(self):
        """Background task for performance monitoring."""
        try:
            while True:
                await asyncio.sleep(60)  # Check every minute
                
                # Update performance metrics
                self.performance_metrics['last_performance_check'] = time.time()
                
                # Log performance summary
                avg_latency = 0.0
                if self.performance_metrics['audit_latency']:
                    avg_latency = sum(self.performance_metrics['audit_latency']) / len(self.performance_metrics['audit_latency'])
                
                if avg_latency > 0.1:  # Alert if audit latency > 100ms
                    logger.warning(f"High audit latency detected: {avg_latency*1000:.1f}ms")
                
        except asyncio.CancelledError:
            logger.info("Performance monitoring loop cancelled")
        except Exception as e:
            logger.error(f"Error in performance monitoring loop: {e}")
    
    async def _cache_cleanup_loop(self):
        """Background task for cache cleanup."""
        try:
            while True:
                await asyncio.sleep(300)  # Cleanup every 5 minutes
                
                # Clean up report cache
                current_time = datetime.now(UTC)
                expired_keys = []
                
                for cache_key, report in self.report_generator.report_cache.items():
                    if current_time - report.generated_at > self.report_generator.cache_expiry:
                        expired_keys.append(cache_key)
                
                for key in expired_keys:
                    del self.report_generator.report_cache[key]
                
                if expired_keys:
                    logger.info(f"Cleaned up {len(expired_keys)} expired report cache entries")
                
        except asyncio.CancelledError:
            logger.info("Cache cleanup loop cancelled")
        except Exception as e:
            logger.error(f"Error in cache cleanup loop: {e}")
    
    async def _register_tool_audit_hooks(self):
        """Register audit hooks with existing tools (placeholder)."""
        # In a real implementation, this would register hooks with all existing tools
        # to automatically audit their execution
        logger.info("Tool audit hooks registration completed")


# Global audit system instance
_audit_system: Optional[AuditSystemManager] = None


def get_audit_system() -> Optional[AuditSystemManager]:
    """Get global audit system instance."""
    return _audit_system


async def initialize_audit_system(config: Optional[AuditConfiguration] = None,
                                compliance_standards: Optional[List[ComplianceStandard]] = None) -> Either[AuditError, AuditSystemManager]:
    """Initialize global audit system."""
    global _audit_system
    
    if _audit_system is None:
        _audit_system = AuditSystemManager(config)
    
    init_result = await _audit_system.initialize(compliance_standards)
    if init_result.is_left():
        return init_result
    
    return Either.right(_audit_system)


async def shutdown_audit_system():
    """Shutdown global audit system."""
    global _audit_system
    
    if _audit_system:
        await _audit_system.shutdown()
        _audit_system = None