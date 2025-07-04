"""
Access Controller - TASK_62 Phase 2 Core Security Engine

Granular access control with context-aware permissions for zero trust security.
Provides dynamic access control, context-aware authorization, and fine-grained permissions.

Architecture: Zero Trust Principles + RBAC + ABAC + Context-Aware Authorization + Dynamic Permissions
Performance: <50ms authorization decision, <100ms permission evaluation, <200ms context analysis
Security: Fail-safe access control, comprehensive audit trail, context-aware decisions
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Tuple, Union, Set
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
from enum import Enum
import asyncio
import json
from pathlib import Path

from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.zero_trust_architecture import (
    TrustScore, PolicyId, SecurityContextId, ThreatId, ValidationId, 
    RiskScore, ComplianceId, TrustLevel, ValidationScope, SecurityOperation,
    AccessDecision, SecurityContext, ZeroTrustError, AccessControlError,
    create_security_context_id, create_trust_score
)


class AccessResult(Enum):
    """Access control decision results."""
    ALLOW = "allow"                  # Access granted
    DENY = "deny"                    # Access denied
    CONDITIONAL = "conditional"      # Access with conditions
    REQUIRES_APPROVAL = "requires_approval"  # Manual approval required
    TEMPORARILY_DENIED = "temporarily_denied"  # Temporarily denied
    ESCALATED = "escalated"          # Escalated for review


class PermissionType(Enum):
    """Types of permissions."""
    READ = "read"                    # Read permission
    WRITE = "write"                  # Write permission
    EXECUTE = "execute"              # Execute permission
    DELETE = "delete"                # Delete permission
    ADMIN = "admin"                  # Administrative permission
    CREATE = "create"                # Create permission
    MODIFY = "modify"                # Modify permission
    VIEW = "view"                    # View permission
    MANAGE = "manage"                # Management permission


class ResourceType(Enum):
    """Types of resources for access control."""
    FILE = "file"                    # File resource
    DIRECTORY = "directory"          # Directory resource
    APPLICATION = "application"      # Application resource
    SERVICE = "service"              # Service resource
    DATABASE = "database"            # Database resource
    API = "api"                      # API resource
    MACRO = "macro"                  # Keyboard Maestro macro
    VARIABLE = "variable"            # Variable resource
    CONFIGURATION = "configuration"  # Configuration resource


class AuthorizationModel(Enum):
    """Authorization models."""
    RBAC = "rbac"                    # Role-Based Access Control
    ABAC = "abac"                    # Attribute-Based Access Control
    DAC = "dac"                      # Discretionary Access Control
    MAC = "mac"                      # Mandatory Access Control
    ZBAC = "zbac"                    # Zone-Based Access Control
    TBAC = "tbac"                    # Task-Based Access Control


@dataclass(frozen=True)
class Permission:
    """Permission specification."""
    permission_id: str
    permission_type: PermissionType
    resource_type: ResourceType
    resource_path: str
    conditions: Dict[str, Any] = field(default_factory=dict)
    constraints: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    granted_by: Optional[str] = None
    granted_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.permission_id or not self.resource_path:
            raise ValueError("Permission ID and resource path are required")
    
    def is_expired(self) -> bool:
        """Check if permission has expired."""
        return self.expires_at is not None and self.expires_at < datetime.now(UTC)
    
    def matches_request(self, resource_path: str, permission_type: PermissionType) -> bool:
        """Check if permission matches access request."""
        # Check permission type
        if self.permission_type != permission_type:
            return False
        
        # Check resource path (supports wildcards)
        if self.resource_path == "*":
            return True
        elif self.resource_path.endswith("/*"):
            base_path = self.resource_path[:-2]
            return resource_path.startswith(base_path)
        else:
            return self.resource_path == resource_path


@dataclass(frozen=True)
class Role:
    """Role specification for RBAC."""
    role_id: str
    role_name: str
    description: str
    permissions: Set[str] = field(default_factory=set)  # Permission IDs
    parent_roles: Set[str] = field(default_factory=set)  # Inherited roles
    conditions: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    created_by: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.role_id or not self.role_name:
            raise ValueError("Role ID and name are required")


@dataclass(frozen=True)
class Subject:
    """Subject (user/service) for access control."""
    subject_id: str
    subject_type: str  # user, service, application
    attributes: Dict[str, Any] = field(default_factory=dict)
    roles: Set[str] = field(default_factory=set)  # Role IDs
    direct_permissions: Set[str] = field(default_factory=set)  # Permission IDs
    groups: Set[str] = field(default_factory=set)
    security_clearance: Optional[str] = None
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_authenticated: Optional[datetime] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.subject_id or not self.subject_type:
            raise ValueError("Subject ID and type are required")


@dataclass(frozen=True)
class AccessRequest:
    """Access request specification."""
    request_id: str
    subject_id: str
    resource_path: str
    resource_type: ResourceType
    permission_type: PermissionType
    context: SecurityContext
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    additional_context: Dict[str, Any] = field(default_factory=dict)
    urgency: str = "normal"  # low, normal, high, critical
    
    def __post_init__(self):
        if not all([self.request_id, self.subject_id, self.resource_path]):
            raise ValueError("Request ID, subject ID, and resource path are required")


@dataclass(frozen=True)
class AuthorizationResult:
    """Authorization decision result."""
    request_id: str
    decision: AccessResult
    subject_id: str
    resource_path: str
    permission_type: PermissionType
    reason: str
    confidence: float  # 0.0 to 1.0
    conditions: List[str] = field(default_factory=list)
    constraints: Dict[str, Any] = field(default_factory=dict)
    expires_at: Optional[datetime] = None
    decided_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    decided_by: str = "access_controller"
    audit_trail: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
    
    def is_expired(self) -> bool:
        """Check if authorization has expired."""
        return self.expires_at is not None and self.expires_at < datetime.now(UTC)


class AccessController:
    """Granular access control with context-aware permissions system."""
    
    def __init__(self):
        self.subjects: Dict[str, Subject] = {}
        self.roles: Dict[str, Role] = {}
        self.permissions: Dict[str, Permission] = {}
        self.authorization_cache: Dict[str, Tuple[AuthorizationResult, datetime]] = {}
        self.access_history: List[AuthorizationResult] = []
        
        # Policy configurations
        self.authorization_model = AuthorizationModel.ABAC  # Default to ABAC
        self.default_deny = True  # Fail-safe default
        self.cache_duration = 300  # 5 minutes
        self.max_cache_size = 10000
        
        # Performance metrics
        self.authorization_count = 0
        self.average_authorization_time = 0.0
        self.cache_hit_rate = 0.0
        self.denial_rate = 0.0
    
    @require(lambda self, subject: isinstance(subject, Subject))
    @ensure(lambda self, result: result.is_right() or isinstance(result.get_left(), AccessControlError))
    async def register_subject(
        self, 
        subject: Subject
    ) -> Either[AccessControlError, str]:
        """Register a subject for access control."""
        try:
            # Validate subject
            if subject.subject_id in self.subjects:
                return Either.left(AccessControlError(
                    f"Subject {subject.subject_id} already exists",
                    "DUPLICATE_SUBJECT",
                    SecurityOperation.AUTHORIZE
                ))
            
            # Validate roles exist
            for role_id in subject.roles:
                if role_id not in self.roles:
                    return Either.left(AccessControlError(
                        f"Role {role_id} does not exist",
                        "INVALID_ROLE",
                        SecurityOperation.AUTHORIZE,
                        {"role_id": role_id}
                    ))
            
            # Validate permissions exist
            for permission_id in subject.direct_permissions:
                if permission_id not in self.permissions:
                    return Either.left(AccessControlError(
                        f"Permission {permission_id} does not exist",
                        "INVALID_PERMISSION",
                        SecurityOperation.AUTHORIZE,
                        {"permission_id": permission_id}
                    ))
            
            # Register subject
            self.subjects[subject.subject_id] = subject
            
            return Either.right(f"Subject {subject.subject_id} registered successfully")
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to register subject: {str(e)}",
                "SUBJECT_REGISTRATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    @require(lambda self, role: isinstance(role, Role))
    async def register_role(
        self, 
        role: Role
    ) -> Either[AccessControlError, str]:
        """Register a role for RBAC."""
        try:
            # Validate role
            if role.role_id in self.roles:
                return Either.left(AccessControlError(
                    f"Role {role.role_id} already exists",
                    "DUPLICATE_ROLE",
                    SecurityOperation.AUTHORIZE
                ))
            
            # Check for circular role inheritance
            if self._has_circular_role_inheritance(role):
                return Either.left(AccessControlError(
                    f"Circular role inheritance detected for role {role.role_id}",
                    "CIRCULAR_ROLE_INHERITANCE",
                    SecurityOperation.AUTHORIZE
                ))
            
            # Validate permissions exist
            for permission_id in role.permissions:
                if permission_id not in self.permissions:
                    return Either.left(AccessControlError(
                        f"Permission {permission_id} does not exist",
                        "INVALID_PERMISSION",
                        SecurityOperation.AUTHORIZE,
                        {"permission_id": permission_id}
                    ))
            
            # Register role
            self.roles[role.role_id] = role
            
            return Either.right(f"Role {role.role_id} registered successfully")
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to register role: {str(e)}",
                "ROLE_REGISTRATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    @require(lambda self, permission: isinstance(permission, Permission))
    async def register_permission(
        self, 
        permission: Permission
    ) -> Either[AccessControlError, str]:
        """Register a permission."""
        try:
            # Validate permission
            if permission.permission_id in self.permissions:
                return Either.left(AccessControlError(
                    f"Permission {permission.permission_id} already exists",
                    "DUPLICATE_PERMISSION",
                    SecurityOperation.AUTHORIZE
                ))
            
            # Validate resource path
            if not permission.resource_path:
                return Either.left(AccessControlError(
                    "Permission must have a resource path",
                    "INVALID_RESOURCE_PATH",
                    SecurityOperation.AUTHORIZE
                ))
            
            # Register permission
            self.permissions[permission.permission_id] = permission
            
            return Either.right(f"Permission {permission.permission_id} registered successfully")
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to register permission: {str(e)}",
                "PERMISSION_REGISTRATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    @require(lambda self, request: isinstance(request, AccessRequest))
    @ensure(lambda self, result: result.is_right() or isinstance(result.get_left(), AccessControlError))
    async def authorize_access(
        self, 
        request: AccessRequest
    ) -> Either[AccessControlError, AuthorizationResult]:
        """Authorize access request with context-aware decision making."""
        try:
            start_time = datetime.now(UTC)
            
            # Check cache first
            cache_key = self._generate_cache_key(request)
            cached_result = self._get_cached_authorization(cache_key)
            if cached_result and not cached_result.is_expired():
                self._update_cache_metrics(True)
                return Either.right(cached_result)
            
            # Validate request
            request_validation = self._validate_access_request(request)
            if request_validation.is_left():
                return request_validation
            
            # Get subject
            if request.subject_id not in self.subjects:
                return Either.left(AccessControlError(
                    f"Subject {request.subject_id} not found",
                    "SUBJECT_NOT_FOUND",
                    SecurityOperation.AUTHORIZE,
                    {"subject_id": request.subject_id}
                ))
            
            subject = self.subjects[request.subject_id]
            
            # Perform authorization based on model
            if self.authorization_model == AuthorizationModel.RBAC:
                auth_result = await self._authorize_rbac(request, subject)
            elif self.authorization_model == AuthorizationModel.ABAC:
                auth_result = await self._authorize_abac(request, subject)
            else:
                # Default to ABAC
                auth_result = await self._authorize_abac(request, subject)
            
            if auth_result.is_left():
                return auth_result
            
            result = auth_result.get_right()
            
            # Apply context-aware modifications
            context_result = await self._apply_context_aware_modifications(result, request)
            if context_result.is_left():
                return context_result
            
            final_result = context_result.get_right()
            
            # Calculate decision confidence
            confidence = self._calculate_decision_confidence(final_result, request, subject)
            
            # Create final authorization result
            end_time = datetime.now(UTC)
            authorization_time = (end_time - start_time).total_seconds() * 1000
            
            authorization_result = AuthorizationResult(
                request_id=request.request_id,
                decision=final_result.decision,
                subject_id=request.subject_id,
                resource_path=request.resource_path,
                permission_type=request.permission_type,
                reason=final_result.reason,
                confidence=confidence,
                conditions=final_result.conditions,
                constraints=final_result.constraints,
                expires_at=self._calculate_authorization_expiration(final_result.decision, request.context),
                audit_trail=[
                    f"Authorization requested at {start_time.isoformat()}",
                    f"Decision: {final_result.decision.value}",
                    f"Authorization time: {authorization_time:.2f}ms",
                    f"Model used: {self.authorization_model.value}"
                ],
                metadata={
                    "authorization_time_ms": authorization_time,
                    "authorization_model": self.authorization_model.value,
                    "cache_hit": False,
                    "trust_level": request.context.trust_level.value,
                    "risk_score": float(request.context.risk_score)
                }
            )
            
            # Cache result
            self._cache_authorization_result(cache_key, authorization_result)
            
            # Store in history
            self.access_history.append(authorization_result)
            
            # Update metrics
            self._update_authorization_metrics(authorization_time, final_result.decision == AccessResult.DENY)
            self._update_cache_metrics(False)
            
            return Either.right(authorization_result)
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Authorization failed: {str(e)}",
                "AUTHORIZATION_ERROR",
                SecurityOperation.AUTHORIZE,
                {"request_id": request.request_id}
            ))
    
    async def revoke_permissions(
        self, 
        subject_id: str, 
        permission_ids: List[str]
    ) -> Either[AccessControlError, str]:
        """Revoke permissions from subject."""
        try:
            if subject_id not in self.subjects:
                return Either.left(AccessControlError(
                    f"Subject {subject_id} not found",
                    "SUBJECT_NOT_FOUND",
                    SecurityOperation.AUTHORIZE,
                    {"subject_id": subject_id}
                ))
            
            subject = self.subjects[subject_id]
            
            # Remove permissions
            updated_permissions = subject.direct_permissions - set(permission_ids)
            
            # Create updated subject
            updated_subject = Subject(
                subject_id=subject.subject_id,
                subject_type=subject.subject_type,
                attributes=subject.attributes,
                roles=subject.roles,
                direct_permissions=updated_permissions,
                groups=subject.groups,
                security_clearance=subject.security_clearance,
                created_at=subject.created_at,
                last_authenticated=subject.last_authenticated,
                metadata=subject.metadata
            )
            
            # Update subject
            self.subjects[subject_id] = updated_subject
            
            # Clear related cache entries
            self._clear_subject_cache(subject_id)
            
            return Either.right(f"Revoked {len(permission_ids)} permissions from subject {subject_id}")
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to revoke permissions: {str(e)}",
                "PERMISSION_REVOCATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    async def get_effective_permissions(
        self, 
        subject_id: str
    ) -> Either[AccessControlError, List[Permission]]:
        """Get effective permissions for subject (direct + role-based)."""
        try:
            if subject_id not in self.subjects:
                return Either.left(AccessControlError(
                    f"Subject {subject_id} not found",
                    "SUBJECT_NOT_FOUND",
                    SecurityOperation.AUTHORIZE,
                    {"subject_id": subject_id}
                ))
            
            subject = self.subjects[subject_id]
            effective_permission_ids = set()
            
            # Add direct permissions
            effective_permission_ids.update(subject.direct_permissions)
            
            # Add role-based permissions
            for role_id in subject.roles:
                role_permissions = self._get_role_permissions_recursive(role_id)
                effective_permission_ids.update(role_permissions)
            
            # Filter out expired permissions
            effective_permissions = []
            for permission_id in effective_permission_ids:
                if permission_id in self.permissions:
                    permission = self.permissions[permission_id]
                    if not permission.is_expired():
                        effective_permissions.append(permission)
            
            return Either.right(effective_permissions)
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to get effective permissions: {str(e)}",
                "EFFECTIVE_PERMISSIONS_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    def _validate_access_request(self, request: AccessRequest) -> Either[AccessControlError, None]:
        """Validate access request."""
        # Check resource path
        if not request.resource_path or request.resource_path.strip() == "":
            return Either.left(AccessControlError(
                "Resource path cannot be empty",
                "INVALID_RESOURCE_PATH",
                SecurityOperation.AUTHORIZE
            ))
        
        # Validate security context
        if not request.context or not request.context.context_id:
            return Either.left(AccessControlError(
                "Valid security context is required",
                "INVALID_CONTEXT",
                SecurityOperation.AUTHORIZE
            ))
        
        return Either.right(None)
    
    async def _authorize_rbac(
        self, 
        request: AccessRequest, 
        subject: Subject
    ) -> Either[AccessControlError, AuthorizationResult]:
        """Perform Role-Based Access Control authorization."""
        try:
            # Get effective permissions
            effective_permissions_result = await self.get_effective_permissions(subject.subject_id)
            if effective_permissions_result.is_left():
                return effective_permissions_result
            
            effective_permissions = effective_permissions_result.get_right()
            
            # Check if any permission matches the request
            matching_permissions = [
                perm for perm in effective_permissions
                if perm.matches_request(request.resource_path, request.permission_type)
            ]
            
            if not matching_permissions:
                # No matching permissions - deny by default
                return Either.right(AuthorizationResult(
                    request_id=request.request_id,
                    decision=AccessResult.DENY,
                    subject_id=request.subject_id,
                    resource_path=request.resource_path,
                    permission_type=request.permission_type,
                    reason="No matching permissions found",
                    confidence=1.0
                ))
            
            # Check permission conditions
            for permission in matching_permissions:
                conditions_met = self._evaluate_permission_conditions(permission, request)
                if conditions_met:
                    return Either.right(AuthorizationResult(
                        request_id=request.request_id,
                        decision=AccessResult.ALLOW,
                        subject_id=request.subject_id,
                        resource_path=request.resource_path,
                        permission_type=request.permission_type,
                        reason=f"Access granted by permission {permission.permission_id}",
                        confidence=1.0,
                        conditions=list(permission.conditions.keys()),
                        constraints=permission.constraints
                    ))
            
            # Permissions found but conditions not met
            return Either.right(AuthorizationResult(
                request_id=request.request_id,
                decision=AccessResult.CONDITIONAL,
                subject_id=request.subject_id,
                resource_path=request.resource_path,
                permission_type=request.permission_type,
                reason="Permission conditions not fully satisfied",
                confidence=0.7,
                conditions=["Review permission conditions"]
            ))
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"RBAC authorization failed: {str(e)}",
                "RBAC_AUTHORIZATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    async def _authorize_abac(
        self, 
        request: AccessRequest, 
        subject: Subject
    ) -> Either[AccessControlError, AuthorizationResult]:
        """Perform Attribute-Based Access Control authorization."""
        try:
            # Evaluate attributes
            decision_factors = []
            
            # Check subject attributes
            subject_score = self._evaluate_subject_attributes(subject, request)
            decision_factors.append(("subject", subject_score))
            
            # Check resource attributes
            resource_score = self._evaluate_resource_attributes(request)
            decision_factors.append(("resource", resource_score))
            
            # Check environmental attributes
            environment_score = self._evaluate_environment_attributes(request)
            decision_factors.append(("environment", environment_score))
            
            # Check action attributes
            action_score = self._evaluate_action_attributes(request)
            decision_factors.append(("action", action_score))
            
            # Calculate overall score
            weights = {"subject": 0.4, "resource": 0.3, "environment": 0.2, "action": 0.1}
            overall_score = sum(
                score * weights.get(factor, 0.25) 
                for factor, score in decision_factors
            )
            
            # Make decision based on score
            if overall_score >= 0.8:
                decision = AccessResult.ALLOW
                reason = "High confidence authorization based on attributes"
            elif overall_score >= 0.6:
                decision = AccessResult.CONDITIONAL
                reason = "Conditional access based on attribute evaluation"
            elif overall_score >= 0.4:
                decision = AccessResult.REQUIRES_APPROVAL
                reason = "Manual approval required based on attribute analysis"
            else:
                decision = AccessResult.DENY
                reason = "Access denied based on attribute evaluation"
            
            return Either.right(AuthorizationResult(
                request_id=request.request_id,
                decision=decision,
                subject_id=request.subject_id,
                resource_path=request.resource_path,
                permission_type=request.permission_type,
                reason=reason,
                confidence=overall_score,
                metadata={
                    "abac_scores": dict(decision_factors),
                    "overall_score": overall_score
                }
            ))
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"ABAC authorization failed: {str(e)}",
                "ABAC_AUTHORIZATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    async def _apply_context_aware_modifications(
        self, 
        initial_result: AuthorizationResult, 
        request: AccessRequest
    ) -> Either[AccessControlError, AuthorizationResult]:
        """Apply context-aware modifications to authorization decision."""
        try:
            modified_decision = initial_result.decision
            modified_conditions = list(initial_result.conditions)
            modified_reason = initial_result.reason
            
            # Check trust level
            if request.context.trust_level == TrustLevel.LOW:
                if modified_decision == AccessResult.ALLOW:
                    modified_decision = AccessResult.CONDITIONAL
                    modified_conditions.append("Additional verification required due to low trust level")
                    modified_reason += " (Modified due to low trust level)"
            elif request.context.trust_level == TrustLevel.UNTRUSTED:
                modified_decision = AccessResult.DENY
                modified_reason = "Access denied due to untrusted context"
            
            # Check risk score
            if request.context.risk_score > 0.7:
                if modified_decision == AccessResult.ALLOW:
                    modified_decision = AccessResult.REQUIRES_APPROVAL
                    modified_reason += " (Escalated due to high risk score)"
            
            # Check time-based restrictions
            current_hour = datetime.now(UTC).hour
            if current_hour < 6 or current_hour > 22:  # Outside business hours
                if modified_decision == AccessResult.ALLOW and request.permission_type in [PermissionType.ADMIN, PermissionType.DELETE]:
                    modified_decision = AccessResult.CONDITIONAL
                    modified_conditions.append("Administrative actions outside business hours require justification")
            
            # Check location-based restrictions
            if request.context.location and "unknown" in request.context.location.lower():
                if modified_decision == AccessResult.ALLOW:
                    modified_decision = AccessResult.CONDITIONAL
                    modified_conditions.append("Location verification required")
            
            return Either.right(AuthorizationResult(
                request_id=initial_result.request_id,
                decision=modified_decision,
                subject_id=initial_result.subject_id,
                resource_path=initial_result.resource_path,
                permission_type=initial_result.permission_type,
                reason=modified_reason,
                confidence=initial_result.confidence,
                conditions=modified_conditions,
                constraints=initial_result.constraints,
                audit_trail=initial_result.audit_trail + ["Applied context-aware modifications"],
                metadata=initial_result.metadata
            ))
            
        except Exception as e:
            return Either.left(AccessControlError(
                f"Failed to apply context-aware modifications: {str(e)}",
                "CONTEXT_MODIFICATION_ERROR",
                SecurityOperation.AUTHORIZE
            ))
    
    def _evaluate_subject_attributes(self, subject: Subject, request: AccessRequest) -> float:
        """Evaluate subject attributes for ABAC."""
        score = 0.5  # Base score
        
        # Check security clearance
        if subject.security_clearance:
            clearance_levels = {"public": 0.3, "internal": 0.5, "confidential": 0.7, "secret": 0.9, "top_secret": 1.0}
            score += clearance_levels.get(subject.security_clearance.lower(), 0.0) * 0.3
        
        # Check authentication recency
        if subject.last_authenticated:
            hours_since_auth = (datetime.now(UTC) - subject.last_authenticated).total_seconds() / 3600
            if hours_since_auth < 1:
                score += 0.2
            elif hours_since_auth < 8:
                score += 0.1
        
        # Check subject type
        if subject.subject_type == "user":
            score += 0.1
        elif subject.subject_type == "service":
            score += 0.2  # Services may be more trusted
        
        return min(1.0, score)
    
    def _evaluate_resource_attributes(self, request: AccessRequest) -> float:
        """Evaluate resource attributes for ABAC."""
        score = 0.5  # Base score
        
        # Check resource type sensitivity
        sensitivity_scores = {
            ResourceType.CONFIGURATION: 0.3,  # High security
            ResourceType.DATABASE: 0.2,
            ResourceType.API: 0.4,
            ResourceType.FILE: 0.7,
            ResourceType.MACRO: 0.6,
            ResourceType.VARIABLE: 0.5
        }
        
        score += sensitivity_scores.get(request.resource_type, 0.5) * 0.4
        
        # Check resource path patterns
        if "/admin/" in request.resource_path.lower():
            score -= 0.3  # Admin resources are more sensitive
        elif "/public/" in request.resource_path.lower():
            score += 0.2  # Public resources less sensitive
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_environment_attributes(self, request: AccessRequest) -> float:
        """Evaluate environmental attributes for ABAC."""
        score = 0.5  # Base score
        
        # Check trust level
        trust_scores = {
            TrustLevel.VERIFIED: 1.0,
            TrustLevel.HIGH: 0.8,
            TrustLevel.MEDIUM: 0.6,
            TrustLevel.LOW: 0.3,
            TrustLevel.UNTRUSTED: 0.0,
            TrustLevel.UNKNOWN: 0.4
        }
        
        score = trust_scores.get(request.context.trust_level, 0.5) * 0.6
        
        # Check risk score
        risk_factor = 1.0 - float(request.context.risk_score)
        score += risk_factor * 0.4
        
        return max(0.0, min(1.0, score))
    
    def _evaluate_action_attributes(self, request: AccessRequest) -> float:
        """Evaluate action attributes for ABAC."""
        # Action risk levels
        action_risks = {
            PermissionType.READ: 0.9,
            PermissionType.VIEW: 0.9,
            PermissionType.WRITE: 0.6,
            PermissionType.CREATE: 0.7,
            PermissionType.MODIFY: 0.5,
            PermissionType.DELETE: 0.2,
            PermissionType.EXECUTE: 0.4,
            PermissionType.ADMIN: 0.1,
            PermissionType.MANAGE: 0.3
        }
        
        return action_risks.get(request.permission_type, 0.5)
    
    def _evaluate_permission_conditions(
        self, 
        permission: Permission, 
        request: AccessRequest
    ) -> bool:
        """Evaluate permission conditions."""
        try:
            for condition_key, condition_value in permission.conditions.items():
                if condition_key == "time_restriction":
                    if not self._check_time_restriction(condition_value):
                        return False
                elif condition_key == "location_restriction":
                    if not self._check_location_restriction(condition_value, request.context):
                        return False
                elif condition_key == "trust_level_minimum":
                    required_level = TrustLevel(condition_value)
                    if not self._meets_trust_level_requirement(required_level, request.context.trust_level):
                        return False
            
            return True
            
        except Exception:
            return False  # Fail safe
    
    def _check_time_restriction(self, time_restriction: Dict[str, Any]) -> bool:
        """Check time-based restrictions."""
        current_time = datetime.now(UTC)
        current_hour = current_time.hour
        
        if "allowed_hours" in time_restriction:
            allowed_hours = time_restriction["allowed_hours"]
            return current_hour in allowed_hours
        
        return True
    
    def _check_location_restriction(
        self, 
        location_restriction: Dict[str, Any], 
        context: SecurityContext
    ) -> bool:
        """Check location-based restrictions."""
        if "allowed_locations" in location_restriction:
            allowed_locations = location_restriction["allowed_locations"]
            return context.location in allowed_locations
        
        if "blocked_locations" in location_restriction:
            blocked_locations = location_restriction["blocked_locations"]
            return context.location not in blocked_locations
        
        return True
    
    def _meets_trust_level_requirement(
        self, 
        required_level: TrustLevel, 
        current_level: TrustLevel
    ) -> bool:
        """Check if current trust level meets requirement."""
        trust_levels = {
            TrustLevel.UNTRUSTED: 0,
            TrustLevel.LOW: 1,
            TrustLevel.MEDIUM: 2,
            TrustLevel.HIGH: 3,
            TrustLevel.VERIFIED: 4,
            TrustLevel.UNKNOWN: 1  # Treat unknown as low
        }
        
        required_value = trust_levels.get(required_level, 0)
        current_value = trust_levels.get(current_level, 0)
        
        return current_value >= required_value
    
    def _calculate_decision_confidence(
        self, 
        result: AuthorizationResult, 
        request: AccessRequest, 
        subject: Subject
    ) -> float:
        """Calculate confidence in authorization decision."""
        base_confidence = result.confidence
        
        # Adjust based on context factors
        confidence_adjustments = 0.0
        
        # Trust level adjustment
        if request.context.trust_level == TrustLevel.VERIFIED:
            confidence_adjustments += 0.1
        elif request.context.trust_level == TrustLevel.UNTRUSTED:
            confidence_adjustments -= 0.3
        
        # Risk score adjustment
        if request.context.risk_score < 0.3:
            confidence_adjustments += 0.1
        elif request.context.risk_score > 0.7:
            confidence_adjustments -= 0.2
        
        # Subject authentication recency
        if subject.last_authenticated:
            hours_since_auth = (datetime.now(UTC) - subject.last_authenticated).total_seconds() / 3600
            if hours_since_auth < 1:
                confidence_adjustments += 0.05
            elif hours_since_auth > 24:
                confidence_adjustments -= 0.1
        
        final_confidence = base_confidence + confidence_adjustments
        return max(0.0, min(1.0, final_confidence))
    
    def _calculate_authorization_expiration(
        self, 
        decision: AccessResult, 
        context: SecurityContext
    ) -> Optional[datetime]:
        """Calculate when authorization expires."""
        if decision == AccessResult.DENY:
            return datetime.now(UTC) + timedelta(minutes=5)  # Short cache for denials
        
        # Base duration based on trust level
        base_durations = {
            TrustLevel.VERIFIED: timedelta(hours=4),
            TrustLevel.HIGH: timedelta(hours=2),
            TrustLevel.MEDIUM: timedelta(hours=1),
            TrustLevel.LOW: timedelta(minutes=30),
            TrustLevel.UNTRUSTED: timedelta(minutes=5),
            TrustLevel.UNKNOWN: timedelta(minutes=15)
        }
        
        duration = base_durations.get(context.trust_level, timedelta(hours=1))
        
        # Adjust based on risk score
        if context.risk_score > 0.7:
            duration = duration / 2  # Reduce duration for high risk
        
        return datetime.now(UTC) + duration
    
    def _get_role_permissions_recursive(self, role_id: str, visited: Set[str] = None) -> Set[str]:
        """Get all permissions for role including inherited ones."""
        if visited is None:
            visited = set()
        
        if role_id in visited or role_id not in self.roles:
            return set()
        
        visited.add(role_id)
        role = self.roles[role_id]
        
        permissions = set(role.permissions)
        
        # Add permissions from parent roles
        for parent_role_id in role.parent_roles:
            parent_permissions = self._get_role_permissions_recursive(parent_role_id, visited)
            permissions.update(parent_permissions)
        
        return permissions
    
    def _has_circular_role_inheritance(self, role: Role, visited: Set[str] = None) -> bool:
        """Check for circular role inheritance."""
        if visited is None:
            visited = set()
        
        if role.role_id in visited:
            return True
        
        visited.add(role.role_id)
        
        for parent_role_id in role.parent_roles:
            if parent_role_id in self.roles:
                parent_role = self.roles[parent_role_id]
                if self._has_circular_role_inheritance(parent_role, visited.copy()):
                    return True
        
        return False
    
    def _generate_cache_key(self, request: AccessRequest) -> str:
        """Generate cache key for authorization request."""
        key_components = [
            request.subject_id,
            request.resource_path,
            request.permission_type.value,
            str(request.context.trust_level.value),
            str(float(request.context.risk_score))
        ]
        
        return ":".join(key_components)
    
    def _get_cached_authorization(self, cache_key: str) -> Optional[AuthorizationResult]:
        """Get cached authorization result."""
        if cache_key in self.authorization_cache:
            result, cached_time = self.authorization_cache[cache_key]
            if (datetime.now(UTC) - cached_time).total_seconds() < self.cache_duration:
                return result
            else:
                # Remove expired cache entry
                del self.authorization_cache[cache_key]
        
        return None
    
    def _cache_authorization_result(
        self, 
        cache_key: str, 
        result: AuthorizationResult
    ) -> None:
        """Cache authorization result."""
        # Clean cache if it's getting too large
        if len(self.authorization_cache) >= self.max_cache_size:
            # Remove oldest entries
            oldest_keys = sorted(
                self.authorization_cache.keys(),
                key=lambda k: self.authorization_cache[k][1]
            )[:self.max_cache_size // 4]
            
            for key in oldest_keys:
                del self.authorization_cache[key]
        
        self.authorization_cache[cache_key] = (result, datetime.now(UTC))
    
    def _clear_subject_cache(self, subject_id: str) -> None:
        """Clear cache entries for specific subject."""
        keys_to_remove = [
            key for key in self.authorization_cache.keys()
            if key.startswith(f"{subject_id}:")
        ]
        
        for key in keys_to_remove:
            del self.authorization_cache[key]
    
    def _update_authorization_metrics(
        self, 
        authorization_time: float, 
        was_denied: bool
    ) -> None:
        """Update authorization performance metrics."""
        self.authorization_count += 1
        
        # Update average authorization time
        if self.authorization_count == 1:
            self.average_authorization_time = authorization_time
        else:
            alpha = 0.1  # Exponential moving average factor
            self.average_authorization_time = (
                alpha * authorization_time + 
                (1 - alpha) * self.average_authorization_time
            )
        
        # Update denial rate
        if self.authorization_count == 1:
            self.denial_rate = 1.0 if was_denied else 0.0
        else:
            alpha = 0.1
            denial_indicator = 1.0 if was_denied else 0.0
            self.denial_rate = (
                alpha * denial_indicator + 
                (1 - alpha) * self.denial_rate
            )
    
    def _update_cache_metrics(self, was_cache_hit: bool) -> None:
        """Update cache performance metrics."""
        total_requests = self.authorization_count + (1 if was_cache_hit else 0)
        
        if total_requests == 1:
            self.cache_hit_rate = 1.0 if was_cache_hit else 0.0
        else:
            alpha = 0.1
            hit_indicator = 1.0 if was_cache_hit else 0.0
            self.cache_hit_rate = (
                alpha * hit_indicator + 
                (1 - alpha) * self.cache_hit_rate
            )


# Utility functions for access control
def create_subject(
    subject_id: str,
    subject_type: str,
    roles: List[str] = None,
    permissions: List[str] = None,
    attributes: Dict[str, Any] = None
) -> Subject:
    """Create a subject with validation."""
    if roles is None:
        roles = []
    if permissions is None:
        permissions = []
    if attributes is None:
        attributes = {}
    
    return Subject(
        subject_id=subject_id,
        subject_type=subject_type,
        attributes=attributes,
        roles=set(roles),
        direct_permissions=set(permissions)
    )


def create_role(
    role_name: str,
    description: str,
    permissions: List[str] = None,
    parent_roles: List[str] = None
) -> Role:
    """Create a role with validation."""
    role_id = f"role_{role_name.lower().replace(' ', '_')}"
    
    if permissions is None:
        permissions = []
    if parent_roles is None:
        parent_roles = []
    
    return Role(
        role_id=role_id,
        role_name=role_name,
        description=description,
        permissions=set(permissions),
        parent_roles=set(parent_roles)
    )


def create_permission(
    permission_type: PermissionType,
    resource_type: ResourceType,
    resource_path: str,
    conditions: Dict[str, Any] = None,
    expires_at: Optional[datetime] = None
) -> Permission:
    """Create a permission with validation."""
    permission_id = f"perm_{permission_type.value}_{resource_type.value}_{hash(resource_path) % 10000}"
    
    if conditions is None:
        conditions = {}
    
    return Permission(
        permission_id=permission_id,
        permission_type=permission_type,
        resource_type=resource_type,
        resource_path=resource_path,
        conditions=conditions,
        expires_at=expires_at
    )


def create_access_request(
    subject_id: str,
    resource_path: str,
    resource_type: ResourceType,
    permission_type: PermissionType,
    context: SecurityContext
) -> AccessRequest:
    """Create an access request with validation."""
    request_id = f"req_{int(datetime.now(UTC).timestamp())}"
    
    return AccessRequest(
        request_id=request_id,
        subject_id=subject_id,
        resource_path=resource_path,
        resource_type=resource_type,
        permission_type=permission_type,
        context=context
    )
