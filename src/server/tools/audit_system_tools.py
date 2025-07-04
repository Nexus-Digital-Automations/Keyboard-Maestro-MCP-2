"""
Advanced audit system MCP tools for enterprise compliance and security monitoring.

This module provides comprehensive audit system tools enabling AI to manage
audit logging, compliance monitoring, report generation, and security analysis
with enterprise-grade capabilities and regulatory compliance support.
"""

from __future__ import annotations
from typing import Dict, Any, Optional, List
from datetime import datetime, timedelta, UTC
import asyncio
import logging

# Context type for MCP operations (optional)
try:
    from mcp import Context
except ImportError:
    Context = None

from ...core.contracts import require, ensure
from ...core.either import Either
from ...core.audit_framework import (
    AuditEventType, ComplianceStandard, AuditConfiguration, 
    RiskLevel, AuditLevel, SecurityLimits
)
from ...audit.audit_system_manager import get_audit_system, initialize_audit_system


logger = logging.getLogger(__name__)


@require(lambda operation: operation in ["log", "query", "report", "monitor", "configure", "status"])
async def km_audit_system(
    operation: str,
    event_type: Optional[str] = None,
    user_id: Optional[str] = None,
    resource_id: Optional[str] = None,
    action_details: Optional[Dict] = None,
    compliance_standard: str = "general",
    time_range: Optional[Dict] = None,
    report_format: str = "json",
    include_sensitive: bool = False,
    audit_level: str = "standard",
    retention_period: int = 365,
    encrypt_logs: bool = True,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Advanced audit system for enterprise compliance and security monitoring.
    
    Operations:
    - log: Log audit events for compliance tracking
    - query: Query audit events with filtering and analysis
    - report: Generate compliance reports for regulatory requirements
    - monitor: Configure and manage compliance monitoring
    - configure: Configure audit system settings and standards
    - status: Get audit system status and performance metrics
    
    Security: Enterprise-grade audit logging with cryptographic integrity
    Performance: <50ms event logging, <2s compliance reports, <100ms queries
    """
    try:
        if ctx:
            await ctx.info(f"Starting audit system operation: {operation}")
        
        # Validate operation
        valid_operations = ["log", "query", "report", "monitor", "configure", "status"]
        if operation not in valid_operations:
            return {
                "success": False,
                "error": {
                    "code": "INVALID_OPERATION",
                    "message": f"Invalid operation '{operation}'. Valid operations: {', '.join(valid_operations)}"
                }
            }
        
        # Initialize audit system if needed
        audit_system = get_audit_system()
        if not audit_system or not audit_system.initialized:
            if ctx:
                await ctx.info("Initializing audit system...")
            
            # Parse compliance standards
            standards = []
            if compliance_standard != "general":
                try:
                    standards = [ComplianceStandard(compliance_standard)]
                except ValueError:
                    return {
                        "success": False,
                        "error": {
                            "code": "INVALID_COMPLIANCE_STANDARD",
                            "message": f"Invalid compliance standard: {compliance_standard}"
                        }
                    }
            
            # Initialize system
            init_result = await initialize_audit_system(compliance_standards=standards)
            if init_result.is_left():
                return {
                    "success": False,
                    "error": {
                        "code": "INITIALIZATION_FAILED",
                        "message": f"Failed to initialize audit system: {init_result.get_left().message}"
                    }
                }
            
            audit_system = init_result.get_right()
        
        # Execute operation
        if operation == "log":
            return await _handle_log_operation(audit_system, event_type, user_id, action_details, ctx)
        elif operation == "query":
            return await _handle_query_operation(audit_system, time_range, action_details, include_sensitive, ctx)
        elif operation == "report":
            return await _handle_report_operation(audit_system, compliance_standard, time_range, report_format, ctx)
        elif operation == "monitor":
            return await _handle_monitor_operation(audit_system, compliance_standard, action_details, ctx)
        elif operation == "configure":
            return await _handle_configure_operation(audit_system, audit_level, retention_period, encrypt_logs, ctx)
        elif operation == "status":
            return await _handle_status_operation(audit_system, ctx)
        else:
            return {
                "success": False,
                "error": {
                    "code": "OPERATION_NOT_IMPLEMENTED",
                    "message": f"Operation '{operation}' not implemented"
                }
            }
            
    except Exception as e:
        logger.error(f"Audit system error: {str(e)}")
        return {
            "success": False,
            "error": {
                "code": "SYSTEM_ERROR",
                "message": f"Audit system operation failed: {str(e)}"
            }
        }


async def _handle_log_operation(audit_system, event_type: Optional[str], user_id: Optional[str], 
                               action_details: Optional[Dict], ctx) -> Dict[str, Any]:
    """Handle audit event logging operation."""
    try:
        if not event_type:
            return {
                "success": False,
                "error": {
                    "code": "MISSING_PARAMETER",
                    "message": "event_type required for log operation"
                }
            }
        
        if not user_id:
            return {
                "success": False,
                "error": {
                    "code": "MISSING_PARAMETER",
                    "message": "user_id required for log operation"
                }
            }
        
        # Parse event type
        try:
            audit_event_type = AuditEventType(event_type)
        except ValueError:
            return {
                "success": False,
                "error": {
                    "code": "INVALID_EVENT_TYPE",
                    "message": f"Invalid event type: {event_type}"
                }
            }
        
        # Extract action details
        details = action_details or {}
        action = details.get('action', f'audit_{event_type}')
        result = details.get('result', 'success')
        session_id = details.get('session_id')
        ip_address = details.get('ip_address')
        risk_level_str = details.get('risk_level', 'low')
        compliance_tags = details.get('compliance_tags', [])
        
        # Parse risk level
        try:
            risk_level = RiskLevel(risk_level_str)
        except ValueError:
            risk_level = RiskLevel.LOW
        
        if ctx:
            await ctx.info(f"Logging audit event: {event_type} for user {user_id}")
        
        # Log the audit event
        log_result = await audit_system.audit_user_action(
            event_type=audit_event_type,
            user_id=user_id,
            action=action,
            result=result,
            session_id=session_id,
            ip_address=ip_address,
            details=details,
            risk_level=risk_level,
            compliance_tags=compliance_tags
        )
        
        if log_result.is_left():
            error = log_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": "AUDIT_LOG_FAILED",
                    "message": f"Failed to log audit event: {error.message}"
                }
            }
        
        event_id = log_result.get_right()
        
        if ctx:
            await ctx.info(f"Audit event logged successfully: {event_id}")
        
        return {
            "success": True,
            "operation": "log",
            "data": {
                "event_id": event_id,
                "event_type": event_type,
                "user_id": user_id,
                "action": action,
                "result": result,
                "timestamp": datetime.now(UTC).isoformat()
            },
            "metadata": {
                "audit_system_status": "operational",
                "compliance_monitoring": "active"
            }
        }
        
    except Exception as e:
        logger.error(f"Error in log operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "LOG_OPERATION_FAILED",
                "message": f"Audit logging failed: {str(e)}"
            }
        }


async def _handle_query_operation(audit_system, time_range: Optional[Dict], filters: Optional[Dict],
                                include_sensitive: bool, ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle audit event query operation."""
    try:
        if ctx:
            await ctx.info("Querying audit events")
        
        # Parse time range
        time_range_tuple = None
        if time_range:
            try:
                start_str = time_range.get('start')
                end_str = time_range.get('end')
                
                if start_str and end_str:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                    time_range_tuple = (start_time, end_time)
                elif start_str:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.now(UTC)
                    time_range_tuple = (start_time, end_time)
                else:
                    # Default to last 24 hours
                    end_time = datetime.now(UTC)
                    start_time = end_time - timedelta(hours=24)
                    time_range_tuple = (start_time, end_time)
            except Exception as e:
                logger.warning(f"Invalid time range format: {e}")
                # Default to last 24 hours
                end_time = datetime.now(UTC)
                start_time = end_time - timedelta(hours=24)
                time_range_tuple = (start_time, end_time)
        else:
            # Default to last 24 hours
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(hours=24)
            time_range_tuple = (start_time, end_time)
        
        # Parse filters
        query_filters = filters or {}
        limit = min(query_filters.get('limit', 100), 1000)  # Max 1000 events
        
        # Query events
        events = await audit_system.query_audit_events(
            filters=query_filters,
            time_range=time_range_tuple,
            limit=limit
        )
        
        # Filter sensitive data if not requested
        if not include_sensitive:
            for event in events:
                # Remove sensitive fields
                if 'details' in event and isinstance(event['details'], dict):
                    sensitive_keys = ['password', 'secret', 'token', 'key', 'credential']
                    for key in list(event['details'].keys()):
                        if any(sensitive_key in key.lower() for sensitive_key in sensitive_keys):
                            event['details'][key] = '[REDACTED]'
        
        if ctx:
            await ctx.info(f"Retrieved {len(events)} audit events")
        
        # Generate summary statistics
        event_types = {}
        risk_levels = {}
        users = set()
        
        for event in events:
            event_type = event.get('event_type', 'unknown')
            risk_level = event.get('risk_level', 'unknown')
            user_id = event.get('user_id', 'unknown')
            
            event_types[event_type] = event_types.get(event_type, 0) + 1
            risk_levels[risk_level] = risk_levels.get(risk_level, 0) + 1
            users.add(user_id)
        
        return {
            "success": True,
            "operation": "query",
            "data": {
                "events": events,
                "summary": {
                    "total_events": len(events),
                    "unique_users": len(users),
                    "event_types": event_types,
                    "risk_levels": risk_levels,
                    "time_range": {
                        "start": time_range_tuple[0].isoformat(),
                        "end": time_range_tuple[1].isoformat()
                    }
                }
            },
            "metadata": {
                "query_time": datetime.now(UTC).isoformat(),
                "sensitive_data_included": include_sensitive,
                "limit_applied": limit
            }
        }
        
    except Exception as e:
        logger.error(f"Error in query operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "QUERY_OPERATION_FAILED", 
                "message": f"Audit query failed: {str(e)}"
            }
        }


async def _handle_report_operation(audit_system, compliance_standard: str, time_range: Optional[Dict],
                                 report_format: str, ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle compliance report generation operation."""
    try:
        if ctx:
            await ctx.info(f"Generating {compliance_standard} compliance report")
        
        # Parse compliance standard
        try:
            standard = ComplianceStandard(compliance_standard)
        except ValueError:
            return {
                "success": False,
                "error": {
                    "code": "INVALID_COMPLIANCE_STANDARD",
                    "message": f"Invalid compliance standard: {compliance_standard}"
                }
            }
        
        # Parse time range (default to last 30 days)
        if time_range:
            try:
                start_str = time_range.get('start')
                end_str = time_range.get('end')
                
                if start_str and end_str:
                    start_time = datetime.fromisoformat(start_str.replace('Z', '+00:00'))
                    end_time = datetime.fromisoformat(end_str.replace('Z', '+00:00'))
                else:
                    end_time = datetime.now(UTC)
                    start_time = end_time - timedelta(days=30)
            except Exception as e:
                logger.warning(f"Invalid time range format: {e}")
                end_time = datetime.now(UTC)
                start_time = end_time - timedelta(days=30)
        else:
            end_time = datetime.now(UTC)
            start_time = end_time - timedelta(days=30)
        
        if ctx:
            await ctx.report_progress(25, 100, "Analyzing compliance data")
        
        # Generate compliance report
        report_result = await audit_system.generate_compliance_report(
            standard=standard,
            period_start=start_time,
            period_end=end_time
        )
        
        if report_result.is_left():
            error = report_result.get_left()
            return {
                "success": False,
                "error": {
                    "code": "REPORT_GENERATION_FAILED",
                    "message": f"Failed to generate compliance report: {error.message}"
                }
            }
        
        report_data = report_result.get_right()
        
        if ctx:
            await ctx.report_progress(75, 100, "Formatting report")
        
        # Format based on requested format
        if report_format.lower() == 'summary':
            # Return just the summary for quick overview
            formatted_data = {
                "report_id": report_data["report_id"],
                "standard": report_data["standard"],
                "period": report_data["period"],
                "summary": report_data["summary"],
                "key_findings": report_data["findings"][:5],  # Top 5 findings
                "top_recommendations": report_data["recommendations"][:3]  # Top 3 recommendations
            }
        else:
            # Return full report
            formatted_data = report_data
        
        if ctx:
            await ctx.report_progress(100, 100, "Report generation complete")
            await ctx.info(f"Report generated: {report_data['summary']['compliance_percentage']:.1f}% compliant")
        
        return {
            "success": True,
            "operation": "report",
            "data": formatted_data,
            "metadata": {
                "report_format": report_format,
                "generation_time": datetime.now(UTC).isoformat(),
                "compliance_score": report_data["summary"]["compliance_percentage"],
                "risk_category": report_data["summary"]["risk_category"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in report operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "REPORT_OPERATION_FAILED",
                "message": f"Compliance report generation failed: {str(e)}"
            }
        }


async def _handle_monitor_operation(audit_system, compliance_standard: str, config: Optional[Dict],
                                  ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle compliance monitoring configuration operation."""
    try:
        if ctx:
            await ctx.info("Configuring compliance monitoring")
        
        monitor = audit_system.compliance_monitor
        config_data = config or {}
        
        # Parse compliance standard for rule loading
        if compliance_standard != "general":
            try:
                standard = ComplianceStandard(compliance_standard)
                monitor.load_standard_rules(standard)
                if ctx:
                    await ctx.info(f"Loaded {standard.value} compliance rules")
            except ValueError:
                return {
                    "success": False,
                    "error": {
                        "code": "INVALID_COMPLIANCE_STANDARD",
                        "message": f"Invalid compliance standard: {compliance_standard}"
                    }
                }
        
        # Handle monitoring commands
        command = config_data.get('command', 'status')
        
        if command == 'enable':
            monitor.enable_monitoring()
            status_message = "Compliance monitoring enabled"
        elif command == 'disable':
            monitor.disable_monitoring()
            status_message = "Compliance monitoring disabled"
        elif command == 'status':
            status_message = f"Compliance monitoring is {'enabled' if monitor.monitoring_enabled else 'disabled'}"
        else:
            status_message = "Monitoring status retrieved"
        
        # Get monitoring statistics
        stats = monitor.get_statistics()
        
        if ctx:
            await ctx.info(status_message)
        
        return {
            "success": True,
            "operation": "monitor",
            "data": {
                "monitoring_enabled": monitor.monitoring_enabled,
                "statistics": stats,
                "status_message": status_message,
                "active_standards": list(set(rule.standard.value for rule in monitor.get_active_rules()))
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "compliance_standard": compliance_standard
            }
        }
        
    except Exception as e:
        logger.error(f"Error in monitor operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "MONITOR_OPERATION_FAILED",
                "message": f"Compliance monitoring configuration failed: {str(e)}"
            }
        }


async def _handle_configure_operation(audit_system, audit_level: str, retention_period: int,
                                    encrypt_logs: bool, ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle audit system configuration operation."""
    try:
        if ctx:
            await ctx.info("Configuring audit system settings")
        
        # Validate audit level
        try:
            level = AuditLevel(audit_level)
        except ValueError:
            return {
                "success": False,
                "error": {
                    "code": "INVALID_AUDIT_LEVEL",
                    "message": f"Invalid audit level: {audit_level}"
                }
            }
        
        # Validate retention period
        if not 1 <= retention_period <= 2555:  # Max 7 years
            return {
                "success": False,
                "error": {
                    "code": "INVALID_RETENTION_PERIOD",
                    "message": f"Invalid retention period: {retention_period} (must be 1-2555 days)"
                }
            }
        
        # Update configuration
        audit_system.config.audit_level = level
        audit_system.config.retention_days = retention_period
        audit_system.config.encrypt_logs = encrypt_logs
        
        # Update event logger configuration
        profile = audit_system.config.get_performance_profile()
        audit_system.event_logger.buffer.max_size = profile['buffer_size']
        audit_system.event_logger.buffer.flush_interval = profile['flush_interval']
        
        if ctx:
            await ctx.info(f"Configuration updated: {audit_level} level, {retention_period} days retention")
        
        return {
            "success": True,
            "operation": "configure",
            "data": {
                "configuration": {
                    "audit_level": audit_level,
                    "retention_period": retention_period,
                    "encryption_enabled": encrypt_logs,
                    "performance_profile": profile
                },
                "status_message": "Audit system configuration updated successfully"
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "configuration_applied": True
            }
        }
        
    except Exception as e:
        logger.error(f"Error in configure operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "CONFIGURE_OPERATION_FAILED",
                "message": f"Audit system configuration failed: {str(e)}"
            }
        }


async def _handle_status_operation(audit_system, ctx: Optional[Context]) -> Dict[str, Any]:
    """Handle audit system status operation."""
    try:
        if ctx:
            await ctx.info("Retrieving audit system status")
        
        # Get comprehensive system status
        status = audit_system.get_system_status()
        
        # Add additional health checks
        health_status = "healthy"
        health_issues = []
        
        # Check performance metrics
        if status.get('performance', {}).get('average_audit_latency_ms', 0) > 100:
            health_issues.append("High audit latency detected")
            health_status = "warning"
        
        # Check error rates
        event_stats = status.get('event_logging', {})
        if event_stats.get('events_failed', 0) > 0:
            failure_rate = (event_stats.get('events_failed', 0) / 
                          max(event_stats.get('events_logged', 1), 1)) * 100
            if failure_rate > 5:
                health_issues.append(f"High failure rate: {failure_rate:.1f}%")
                health_status = "warning"
        
        # Check compliance monitoring
        monitor_stats = status.get('compliance_monitoring', {})
        if not monitor_stats.get('monitoring_enabled', False):
            health_issues.append("Compliance monitoring disabled")
            if health_status == "healthy":
                health_status = "warning"
        
        if not health_issues:
            health_issues.append("All systems operational")
        
        if ctx:
            await ctx.info(f"Audit system status: {health_status}")
        
        return {
            "success": True,
            "operation": "status",
            "data": {
                "system_status": status,
                "health": {
                    "status": health_status,
                    "issues": health_issues,
                    "last_check": datetime.now(UTC).isoformat()
                },
                "capabilities": {
                    "event_logging": True,
                    "compliance_monitoring": True,
                    "report_generation": True,
                    "encryption": status.get('configuration', {}).get('encryption_enabled', False),
                    "real_time_monitoring": True
                }
            },
            "metadata": {
                "timestamp": datetime.now(UTC).isoformat(),
                "version": "1.0.0",
                "features": ["enterprise_compliance", "real_time_monitoring", "automated_reporting"]
            }
        }
        
    except Exception as e:
        logger.error(f"Error in status operation: {e}")
        return {
            "success": False,
            "error": {
                "code": "STATUS_OPERATION_FAILED",
                "message": f"Audit system status retrieval failed: {str(e)}"
            }
        }