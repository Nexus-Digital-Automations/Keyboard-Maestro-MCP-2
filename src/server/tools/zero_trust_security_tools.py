"""
Zero Trust Security Tools - TASK_62 Phase 3 MCP Tools Implementation

FastMCP tools for zero trust security validation, policy enforcement, monitoring, and access control.
Provides Claude Desktop integration for comprehensive zero trust security management.

Architecture: FastMCP Protocol + Zero Trust Architecture + Security Validation + Policy Enforcement
Performance: <200ms tool execution, <100ms validation, <300ms policy enforcement
Integration: Complete zero trust security framework with MCP tools for Claude Desktop
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Annotated, Union
from datetime import datetime, UTC
import asyncio
import json
from pathlib import Path

# FastMCP imports
from fastmcp import FastMCP, Context
from pydantic import Field

# Core imports
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.zero_trust_architecture import (
    TrustScore, PolicyId, SecurityContextId, ThreatId, ValidationId,
    RiskScore, ComplianceId, TrustLevel, ValidationScope, SecurityOperation,
    PolicyType, EnforcementMode, ThreatSeverity, ComplianceFramework,
    TrustValidationResult, SecurityPolicyResult, SecurityMonitoringResult,
    AccessDecision, SecurityContext, ZeroTrustError, SecurityValidationError,
    create_trust_score, create_security_context_id, create_policy_id
)

# Security engine imports
from src.security.trust_validator import ZeroTrustValidator
from src.security.policy_enforcer import SecurityPolicyEnforcer
from src.security.security_monitor import SecurityMonitor
from src.security.access_controller import AccessController

# Initialize FastMCP server for zero trust security tools
mcp = FastMCP(
    name="ZeroTrustSecurityTools",
    instructions="Zero trust security framework providing continuous validation, policy enforcement, monitoring, and access control for enterprise automation platforms."
)

# Initialize security engine components
trust_validator = ZeroTrustValidator()
policy_enforcer = SecurityPolicyEnforcer() 
security_monitor = SecurityMonitor()
access_controller = AccessController()


@mcp.tool()
async def km_validate_trust(
    validation_scope: Annotated[str, Field(description="Validation scope (user|device|application|network)")],
    target_id: Annotated[str, Field(description="Target identifier for validation")],
    validation_criteria: Annotated[List[str], Field(description="Trust validation criteria")] = ["identity", "device", "location", "behavior"],
    trust_level_required: Annotated[str, Field(description="Required trust level (low|medium|high|critical)")] = "medium",
    continuous_validation: Annotated[bool, Field(description="Enable continuous trust validation")] = True,
    risk_tolerance: Annotated[str, Field(description="Risk tolerance level (strict|balanced|permissive)")] = "balanced",
    include_context: Annotated[bool, Field(description="Include contextual factors in validation")] = True,
    generate_trust_score: Annotated[bool, Field(description="Generate quantitative trust score")] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Perform continuous trust validation and verification using zero trust principles.
    
    FastMCP Tool for trust validation through Claude Desktop.
    Validates identity, device, location, and behavioral factors for zero trust security.
    
    Returns trust validation results, scores, risk assessment, and remediation recommendations.
    """
    try:
        if ctx:
            await ctx.info(f"Starting zero trust validation for {validation_scope}: {target_id}")
        
        # Parse validation scope
        try:
            scope = ValidationScope(validation_scope.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid validation scope: {validation_scope}. Must be one of: user, device, application, network"
            }
        
        # Parse trust level
        try:
            required_level = TrustLevel(trust_level_required.upper())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid trust level: {trust_level_required}. Must be one of: low, medium, high, critical"
            }
        
        # Perform trust validation
        validation_result = await trust_validator.validate_trust(
            scope=scope,
            target_id=target_id,
            criteria=validation_criteria,
            continuous=continuous_validation,
            include_context=include_context
        )
        
        if validation_result.is_error():
            error_msg = str(validation_result.error)
            if ctx:
                await ctx.error(f"Trust validation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "validation_id": None
            }
        
        result = validation_result.value
        
        # Check if trust level meets requirements
        trust_sufficient = result.trust_level.value >= required_level.value
        
        # Generate trust score if requested
        trust_score = None
        if generate_trust_score:
            score_result = await trust_validator.calculate_composite_trust_score(
                target_id=target_id,
                factors=validation_criteria
            )
            if score_result.is_success():
                trust_score = float(score_result.value)
        
        # Generate risk assessment
        risk_factors = result.risk_factors or []
        risk_level = "low" if trust_sufficient else "high"
        if len(risk_factors) > 3:
            risk_level = "critical"
        elif len(risk_factors) > 1:
            risk_level = "medium"
        
        # Generate recommendations
        recommendations = result.recommendations or []
        if not trust_sufficient:
            recommendations.append(f"Trust level {result.trust_level.value} below required {required_level.value}")
            recommendations.append("Consider additional authentication factors")
        
        if ctx:
            await ctx.info(f"Trust validation completed - Level: {result.trust_level.value}, Score: {trust_score}")
        
        return {
            "success": True,
            "validation_id": result.validation_id,
            "target_id": target_id,
            "validation_scope": validation_scope,
            "trust_level": result.trust_level.value,
            "trust_level_sufficient": trust_sufficient,
            "trust_score": trust_score,
            "validation_timestamp": result.validation_timestamp.isoformat(),
            "criteria_results": result.criteria_results,
            "risk_factors": risk_factors,
            "risk_level": risk_level,
            "recommendations": recommendations,
            "continuous_validation": continuous_validation,
            "expires_at": result.expires_at.isoformat() if result.expires_at else None,
            "metadata": result.metadata
        }
        
    except Exception as e:
        error_msg = f"Trust validation error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "validation_id": None
        }


@mcp.tool()
async def km_enforce_security_policy(
    policy_name: Annotated[str, Field(description="Security policy name or ID")],
    enforcement_scope: Annotated[str, Field(description="Enforcement scope (user|group|application|system)")],
    target_resources: Annotated[List[str], Field(description="Target resources for policy enforcement")],
    enforcement_mode: Annotated[str, Field(description="Enforcement mode (monitor|warn|block|remediate)")] = "warn",
    policy_parameters: Annotated[Optional[Dict[str, Any]], Field(description="Policy-specific parameters")] = None,
    exceptions: Annotated[Optional[List[str]], Field(description="Policy exceptions or exemptions")] = None,
    audit_enforcement: Annotated[bool, Field(description="Audit policy enforcement actions")] = True,
    real_time_enforcement: Annotated[bool, Field(description="Enable real-time policy enforcement")] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Enforce dynamic security policies with real-time monitoring and compliance tracking.
    
    FastMCP Tool for security policy enforcement through Claude Desktop.
    Implements and enforces security policies with configurable enforcement modes.
    
    Returns enforcement results, compliance status, violations, and audit information.
    """
    try:
        if ctx:
            await ctx.info(f"Starting policy enforcement: {policy_name} for scope: {enforcement_scope}")
        
        # Parse enforcement mode
        try:
            mode = EnforcementMode(enforcement_mode.lower())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid enforcement mode: {enforcement_mode}. Must be one of: monitor, warn, block, remediate"
            }
        
        # Create policy ID
        policy_id = create_policy_id(policy_name)
        
        # Prepare policy configuration
        policy_config = {
            "policy_id": policy_id,
            "enforcement_scope": enforcement_scope,
            "target_resources": target_resources,
            "enforcement_mode": mode,
            "parameters": policy_parameters or {},
            "exceptions": exceptions or [],
            "real_time": real_time_enforcement
        }
        
        # Enforce policy
        enforcement_result = await policy_enforcer.enforce_policy(
            policy_id=policy_id,
            scope=enforcement_scope,
            resources=target_resources,
            mode=mode,
            parameters=policy_parameters or {},
            exceptions=exceptions or []
        )
        
        if enforcement_result.is_error():
            error_msg = str(enforcement_result.error)
            if ctx:
                await ctx.error(f"Policy enforcement failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "policy_id": policy_id
            }
        
        result = enforcement_result.value
        
        # Evaluate compliance
        compliance_result = await policy_enforcer.evaluate_compliance(
            policy_id=policy_id,
            framework=ComplianceFramework.SOC2,  # Default to SOC2
            scope=enforcement_scope
        )
        
        compliance_status = "compliant"
        compliance_details = {}
        if compliance_result.is_success():
            compliance_data = compliance_result.value
            compliance_status = "compliant" if compliance_data.get("compliant", False) else "non_compliant"
            compliance_details = compliance_data
        
        # Generate audit record if requested
        audit_record = None
        if audit_enforcement:
            audit_record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "policy_id": policy_id,
                "enforcement_action": result.enforcement_action,
                "affected_resources": result.affected_resources,
                "violations_detected": len(result.violations),
                "compliance_status": compliance_status
            }
        
        if ctx:
            await ctx.info(f"Policy enforcement completed - Action: {result.enforcement_action}, Violations: {len(result.violations)}")
        
        return {
            "success": True,
            "policy_id": policy_id,
            "policy_name": policy_name,
            "enforcement_scope": enforcement_scope,
            "enforcement_mode": enforcement_mode,
            "enforcement_action": result.enforcement_action,
            "enforcement_timestamp": result.enforcement_timestamp.isoformat(),
            "affected_resources": result.affected_resources,
            "violations": [
                {
                    "resource": v.resource,
                    "violation_type": v.violation_type,
                    "severity": v.severity,
                    "description": v.description,
                    "remediation": v.remediation_action
                }
                for v in result.violations
            ],
            "compliance_status": compliance_status,
            "compliance_details": compliance_details,
            "audit_record": audit_record,
            "metadata": result.metadata
        }
        
    except Exception as e:
        error_msg = f"Policy enforcement error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "policy_id": policy_name
        }


@mcp.tool()
async def km_monitor_security_posture(
    monitoring_scope: Annotated[str, Field(description="Monitoring scope (system|application|network|user)")],
    monitoring_duration: Annotated[int, Field(description="Monitoring duration in minutes")] = 60,
    threat_detection: Annotated[bool, Field(description="Enable threat detection and analysis")] = True,
    incident_response: Annotated[bool, Field(description="Enable automated incident response")] = False,
    compliance_monitoring: Annotated[bool, Field(description="Monitor compliance status")] = True,
    alert_thresholds: Annotated[Optional[Dict[str, Any]], Field(description="Custom alert thresholds")] = None,
    include_metrics: Annotated[bool, Field(description="Include security metrics and analytics")] = True,
    real_time_updates: Annotated[bool, Field(description="Provide real-time security updates")] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Monitor real-time security posture with threat detection and compliance tracking.
    
    FastMCP Tool for security monitoring through Claude Desktop.
    Provides comprehensive security monitoring, threat detection, and incident response.
    
    Returns security status, threat analysis, incidents, and compliance information.
    """
    try:
        if ctx:
            await ctx.info(f"Starting security posture monitoring for scope: {monitoring_scope}")
        
        # Start security monitoring
        monitoring_result = await security_monitor.start_monitoring(
            scope=monitoring_scope,
            duration_minutes=monitoring_duration,
            enable_threat_detection=threat_detection,
            enable_incident_response=incident_response
        )
        
        if monitoring_result.is_error():
            error_msg = str(monitoring_result.error)
            if ctx:
                await ctx.error(f"Security monitoring failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "monitoring_session": None
            }
        
        session_data = monitoring_result.value
        
        # Process security events
        events_result = await security_monitor.process_security_events(
            scope=monitoring_scope,
            time_window_minutes=min(monitoring_duration, 15)  # Process recent events
        )
        
        security_events = []
        if events_result.is_success():
            events_data = events_result.value
            security_events = events_data.get("events", [])
        
        # Detect threats if enabled
        threats_detected = []
        if threat_detection:
            threat_result = await security_monitor.detect_threats(
                scope=monitoring_scope,
                events=security_events
            )
            if threat_result.is_success():
                threats_detected = threat_result.value.get("threats", [])
        
        # Check compliance if enabled
        compliance_status = {}
        if compliance_monitoring:
            compliance_result = await security_monitor.assess_compliance(
                scope=monitoring_scope,
                frameworks=[ComplianceFramework.SOC2, ComplianceFramework.ISO27001]
            )
            if compliance_result.is_success():
                compliance_status = compliance_result.value
        
        # Generate security metrics if enabled
        security_metrics = {}
        if include_metrics:
            metrics_result = await security_monitor.get_security_metrics(
                scope=monitoring_scope,
                time_period_minutes=monitoring_duration
            )
            if metrics_result.is_success():
                security_metrics = metrics_result.value
        
        # Calculate overall security score
        security_score = 100.0  # Start with perfect score
        if threats_detected:
            security_score -= len(threats_detected) * 10  # Reduce for threats
        if security_events:
            security_score -= len([e for e in security_events if e.get("severity") == "high"]) * 5
        security_score = max(0.0, security_score)  # Don't go below 0
        
        # Determine overall status
        if security_score >= 90:
            overall_status = "excellent"
        elif security_score >= 75:
            overall_status = "good"
        elif security_score >= 60:
            overall_status = "fair"
        else:
            overall_status = "poor"
        
        if ctx:
            await ctx.info(f"Security monitoring completed - Score: {security_score}, Status: {overall_status}")
        
        return {
            "success": True,
            "monitoring_session": session_data.get("session_id"),
            "monitoring_scope": monitoring_scope,
            "monitoring_duration": monitoring_duration,
            "start_time": session_data.get("start_time", datetime.now(UTC).isoformat()),
            "security_score": security_score,
            "overall_status": overall_status,
            "security_events": [
                {
                    "event_id": event.get("event_id"),
                    "event_type": event.get("event_type"),
                    "severity": event.get("severity"),
                    "timestamp": event.get("timestamp"),
                    "description": event.get("description")
                }
                for event in security_events[:10]  # Limit to recent events
            ],
            "threats_detected": [
                {
                    "threat_id": threat.get("threat_id"),
                    "threat_type": threat.get("threat_type"),
                    "severity": threat.get("severity"),
                    "confidence": threat.get("confidence"),
                    "description": threat.get("description"),
                    "mitigation": threat.get("mitigation")
                }
                for threat in threats_detected
            ],
            "compliance_status": compliance_status,
            "security_metrics": security_metrics,
            "alerts_configured": alert_thresholds is not None,
            "real_time_monitoring": real_time_updates
        }
        
    except Exception as e:
        error_msg = f"Security monitoring error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "monitoring_session": None
        }


@mcp.tool()
async def km_manage_access_control(
    operation: Annotated[str, Field(description="Access control operation (grant|revoke|check|list)")],
    subject_id: Annotated[str, Field(description="Subject identifier (user, service, application)")],
    resource_path: Annotated[str, Field(description="Resource path or identifier")],
    permission_type: Annotated[str, Field(description="Permission type (read|write|execute|delete|admin)")] = "read",
    resource_type: Annotated[str, Field(description="Resource type (file|directory|application|service|macro)")] = "file",
    authorization_model: Annotated[str, Field(description="Authorization model (rbac|abac|dac|mac)")] = "rbac",
    conditions: Annotated[Optional[Dict[str, Any]], Field(description="Access conditions and constraints")] = None,
    temporary_access: Annotated[bool, Field(description="Grant temporary access with expiration")] = False,
    access_duration_hours: Annotated[int, Field(description="Access duration in hours for temporary access")] = 24,
    audit_access: Annotated[bool, Field(description="Audit access control operations")] = True,
    ctx: Context = None
) -> Dict[str, Any]:
    """
    Manage granular access control with context-aware permissions and dynamic authorization.
    
    FastMCP Tool for access control management through Claude Desktop.
    Provides comprehensive access control operations with RBAC, ABAC, and context-aware authorization.
    
    Returns access control results, permissions, audit information, and authorization details.
    """
    try:
        if ctx:
            await ctx.info(f"Starting access control operation: {operation} for {subject_id}")
        
        # Validate operation
        valid_operations = ["grant", "revoke", "check", "list"]
        if operation.lower() not in valid_operations:
            return {
                "success": False,
                "error": f"Invalid operation: {operation}. Must be one of: {valid_operations}"
            }
        
        # Parse permission type
        from src.security.access_controller import PermissionType, ResourceType, AuthorizationModel
        
        try:
            perm_type = PermissionType(permission_type.upper())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid permission type: {permission_type}"
            }
        
        try:
            res_type = ResourceType(resource_type.upper())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid resource type: {resource_type}"
            }
        
        try:
            auth_model = AuthorizationModel(authorization_model.upper())
        except ValueError:
            return {
                "success": False,
                "error": f"Invalid authorization model: {authorization_model}"
            }
        
        # Execute access control operation
        if operation.lower() == "grant":
            # Grant access
            expiration = None
            if temporary_access:
                from datetime import timedelta
                expiration = datetime.now(UTC) + timedelta(hours=access_duration_hours)
            
            result = await access_controller.grant_access(
                subject_id=subject_id,
                resource_path=resource_path,
                permission_type=perm_type,
                resource_type=res_type,
                conditions=conditions or {},
                expires_at=expiration
            )
            
        elif operation.lower() == "revoke":
            # Revoke access
            result = await access_controller.revoke_access(
                subject_id=subject_id,
                resource_path=resource_path,
                permission_type=perm_type
            )
            
        elif operation.lower() == "check":
            # Check access
            result = await access_controller.check_access(
                subject_id=subject_id,
                resource_path=resource_path,
                permission_type=perm_type,
                resource_type=res_type,
                authorization_model=auth_model,
                context=conditions or {}
            )
            
        elif operation.lower() == "list":
            # List permissions
            result = await access_controller.list_permissions(
                subject_id=subject_id,
                resource_path=resource_path if resource_path != "*" else None
            )
        
        else:
            return {
                "success": False,
                "error": f"Unsupported operation: {operation}"
            }
        
        if result.is_error():
            error_msg = str(result.error)
            if ctx:
                await ctx.error(f"Access control operation failed: {error_msg}")
            return {
                "success": False,
                "error": error_msg,
                "operation": operation
            }
        
        result_data = result.value
        
        # Generate audit record if requested
        audit_record = None
        if audit_access:
            audit_record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "operation": operation,
                "subject_id": subject_id,
                "resource_path": resource_path,
                "permission_type": permission_type,
                "authorization_model": authorization_model,
                "result": "success",
                "temporary_access": temporary_access
            }
        
        if ctx:
            await ctx.info(f"Access control operation completed: {operation} - Result: success")
        
        # Format response based on operation type
        if operation.lower() == "check":
            return {
                "success": True,
                "operation": operation,
                "subject_id": subject_id,
                "resource_path": resource_path,
                "permission_type": permission_type,
                "access_granted": result_data.get("access_decision") == "allow",
                "access_decision": result_data.get("access_decision"),
                "decision_reason": result_data.get("decision_reason"),
                "authorization_model": authorization_model,
                "context_factors": result_data.get("context_factors", {}),
                "audit_record": audit_record
            }
            
        elif operation.lower() == "list":
            return {
                "success": True,
                "operation": operation,
                "subject_id": subject_id,
                "permissions": result_data.get("permissions", []),
                "total_permissions": len(result_data.get("permissions", [])),
                "audit_record": audit_record
            }
            
        else:  # grant or revoke
            return {
                "success": True,
                "operation": operation,
                "subject_id": subject_id,
                "resource_path": resource_path,
                "permission_type": permission_type,
                "resource_type": resource_type,
                "temporary_access": temporary_access,
                "expires_at": result_data.get("expires_at"),
                "operation_timestamp": result_data.get("timestamp", datetime.now(UTC).isoformat()),
                "audit_record": audit_record
            }
        
    except Exception as e:
        error_msg = f"Access control management error: {str(e)}"
        if ctx:
            await ctx.error(error_msg)
        return {
            "success": False,
            "error": error_msg,
            "operation": operation
        }


# Export the FastMCP server instance
__all__ = ["mcp"]