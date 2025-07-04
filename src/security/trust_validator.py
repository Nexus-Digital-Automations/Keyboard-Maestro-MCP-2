"""
Trust Validator - TASK_62 Phase 2 Core Security Engine

Continuous trust validation and verification system for zero trust security.
Provides real-time trust assessment, validation criteria evaluation, and continuous monitoring.

Architecture: Zero Trust Principles + Design by Contract + Continuous Validation + Risk Assessment
Performance: <100ms trust validation, <50ms score calculation, <200ms comprehensive assessment
Security: Never trust always verify, continuous validation, comprehensive threat detection
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
    TrustValidationResult, SecurityContext, ValidationCriteria,
    ZeroTrustError, TrustValidationError, SecurityMonitoringError,
    create_trust_score, create_validation_id, create_risk_score,
    determine_trust_level, calculate_composite_trust_score,
    validate_security_context, validate_trust_result
)


class ValidationStatus(Enum):
    """Trust validation status."""
    PENDING = "pending"              # Validation in progress
    COMPLETED = "completed"          # Validation completed successfully
    FAILED = "failed"                # Validation failed
    EXPIRED = "expired"              # Validation result expired
    INVALID = "invalid"              # Validation data invalid
    REQUIRES_REVIEW = "requires_review"  # Manual review required


class TrustFactor(Enum):
    """Trust validation factors."""
    IDENTITY = "identity"            # Identity verification
    DEVICE = "device"                # Device trust and compliance
    LOCATION = "location"            # Location verification
    BEHAVIOR = "behavior"            # Behavioral analysis
    NETWORK = "network"              # Network security assessment
    TEMPORAL = "temporal"            # Time-based validation
    CREDENTIAL = "credential"        # Credential verification
    REPUTATION = "reputation"        # Reputation scoring


@dataclass(frozen=True)
class TrustValidationRequest:
    """Trust validation request specification."""
    validation_id: ValidationId
    scope: ValidationScope
    target_id: str
    context: SecurityContext
    criteria: ValidationCriteria
    requested_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    priority: str = "normal"  # low, normal, high, critical
    timeout: int = 30  # seconds
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def __post_init__(self):
        if not self.target_id:
            raise ValueError("Target ID cannot be empty")
        if self.timeout <= 0:
            raise ValueError("Timeout must be positive")


@dataclass(frozen=True)
class TrustFactorResult:
    """Individual trust factor validation result."""
    factor: TrustFactor
    score: TrustScore
    confidence: float  # 0.0 to 1.0
    evidence: Dict[str, Any]
    warnings: List[str] = field(default_factory=list)
    validation_time: float = 0.0  # milliseconds
    status: ValidationStatus = ValidationStatus.COMPLETED
    
    def __post_init__(self):
        if not (0.0 <= self.confidence <= 1.0):
            raise ValueError("Confidence must be between 0.0 and 1.0")
        if self.validation_time < 0:
            raise ValueError("Validation time cannot be negative")


@dataclass(frozen=True)
class ContinuousValidationConfig:
    """Configuration for continuous trust validation."""
    enabled: bool = True
    validation_interval: int = 300  # seconds
    score_threshold: float = 0.7
    auto_remediation: bool = False
    notification_threshold: float = 0.5
    max_validation_age: int = 3600  # seconds
    factors_to_monitor: Set[TrustFactor] = field(default_factory=lambda: {
        TrustFactor.IDENTITY, TrustFactor.DEVICE, TrustFactor.BEHAVIOR
    })
    escalation_rules: Dict[str, Any] = field(default_factory=dict)


class TrustValidator:
    """Continuous trust validation and verification system."""
    
    def __init__(self):
        self.active_validations: Dict[ValidationId, TrustValidationRequest] = {}
        self.validation_results: Dict[ValidationId, TrustValidationResult] = {}
        self.continuous_configs: Dict[str, ContinuousValidationConfig] = {}
        self.trust_cache: Dict[str, Tuple[TrustScore, datetime]] = {}
        self.validation_history: List[TrustValidationResult] = []
        self.threat_indicators: Dict[str, List[str]] = {}
        
        # Performance metrics
        self.validation_count = 0
        self.average_validation_time = 0.0
        self.cache_hit_rate = 0.0
    
    @require(lambda self, request: isinstance(request, TrustValidationRequest))
    @ensure(lambda self, result: result.is_right() or isinstance(result.get_left(), TrustValidationError))
    async def validate_trust(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustValidationResult]:
        """Perform comprehensive trust validation."""
        try:
            start_time = datetime.now(UTC)
            
            # Validate request
            request_validation = self._validate_request(request)
            if request_validation.is_left():
                return request_validation
            
            # Check cache first
            cached_result = self._get_cached_trust_score(request.target_id, request.scope)
            if cached_result:
                trust_score, cached_time = cached_result
                if (start_time - cached_time).total_seconds() < 300:  # 5 minutes cache
                    return self._create_cached_validation_result(request, trust_score, cached_time)
            
            # Store active validation
            self.active_validations[request.validation_id] = request
            
            # Perform factor validations
            factor_results = await self._validate_trust_factors(request)
            if factor_results.is_left():
                return factor_results
            
            factors = factor_results.get_right()
            
            # Calculate composite trust score
            composite_score = self._calculate_composite_trust_score(factors)
            trust_level = determine_trust_level(composite_score)
            
            # Assess risk factors
            risk_factors = self._assess_risk_factors(request, factors)
            
            # Generate recommendations
            recommendations = self._generate_trust_recommendations(factors, trust_level)
            
            # Calculate expiration
            expires_at = self._calculate_validation_expiration(trust_level, request.criteria)
            
            # Create validation result
            end_time = datetime.now(UTC)
            validation_duration = (end_time - start_time).total_seconds() * 1000  # milliseconds
            
            criteria_results = {
                factor.value: result.status == ValidationStatus.COMPLETED and result.score >= 0.5
                for factor, result in factors.items()
            }
            
            validation_result = TrustValidationResult(
                validation_id=request.validation_id,
                scope=request.scope,
                target_id=request.target_id,
                trust_score=composite_score,
                trust_level=trust_level,
                validation_timestamp=start_time,
                criteria_results=criteria_results,
                risk_factors=risk_factors,
                recommendations=recommendations,
                expires_at=expires_at,
                metadata={
                    "validation_duration_ms": validation_duration,
                    "factor_count": len(factors),
                    "cache_hit": False,
                    "validation_version": "1.0"
                }
            )
            
            # Validate result integrity
            if not validate_trust_result(validation_result):
                return Either.left(TrustValidationError(
                    "Trust validation result failed integrity check",
                    "INVALID_VALIDATION_RESULT",
                    SecurityOperation.VALIDATE
                ))
            
            # Store result and update cache
            self.validation_results[request.validation_id] = validation_result
            self.trust_cache[f"{request.target_id}:{request.scope.value}"] = (composite_score, start_time)
            self.validation_history.append(validation_result)
            
            # Clean up
            if request.validation_id in self.active_validations:
                del self.active_validations[request.validation_id]
            
            # Update metrics
            self._update_validation_metrics(validation_duration)
            
            return Either.right(validation_result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Trust validation failed: {str(e)}",
                "VALIDATION_ERROR",
                SecurityOperation.VALIDATE,
                {"target_id": request.target_id, "scope": request.scope.value}
            ))
    
    @require(lambda self, target_id, scope: target_id and isinstance(scope, ValidationScope))
    async def get_current_trust_score(
        self, 
        target_id: str, 
        scope: ValidationScope
    ) -> Either[TrustValidationError, TrustScore]:
        """Get current trust score for target."""
        try:
            # Check cache first
            cache_key = f"{target_id}:{scope.value}"
            if cache_key in self.trust_cache:
                trust_score, cached_time = self.trust_cache[cache_key]
                # Return cached score if recent (within 10 minutes)
                if (datetime.now(UTC) - cached_time).total_seconds() < 600:
                    return Either.right(trust_score)
            
            # Find most recent validation result
            recent_result = None
            for result in reversed(self.validation_history):
                if result.target_id == target_id and result.scope == scope:
                    if not result.expires_at or result.expires_at > datetime.now(UTC):
                        recent_result = result
                        break
            
            if recent_result:
                return Either.right(recent_result.trust_score)
            
            # No valid trust score found
            return Either.left(TrustValidationError(
                f"No valid trust score found for target {target_id}",
                "NO_TRUST_SCORE",
                SecurityOperation.VALIDATE,
                {"target_id": target_id, "scope": scope.value}
            ))
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Failed to get trust score: {str(e)}",
                "TRUST_SCORE_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def setup_continuous_validation(
        self, 
        target_id: str, 
        config: ContinuousValidationConfig
    ) -> Either[TrustValidationError, str]:
        """Setup continuous trust validation for target."""
        try:
            # Validate configuration
            if config.validation_interval < 60:  # Minimum 1 minute
                return Either.left(TrustValidationError(
                    "Continuous validation interval must be at least 60 seconds",
                    "INVALID_INTERVAL",
                    SecurityOperation.VALIDATE
                ))
            
            # Store configuration
            self.continuous_configs[target_id] = config
            
            # Start continuous validation if enabled
            if config.enabled:
                asyncio.create_task(self._run_continuous_validation(target_id, config))
            
            return Either.right(f"Continuous validation configured for {target_id}")
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Failed to setup continuous validation: {str(e)}",
                "CONTINUOUS_SETUP_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_trust_factors(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, Dict[TrustFactor, TrustFactorResult]]:
        """Validate individual trust factors."""
        try:
            factor_results = {}
            
            # Validate identity factor
            if request.criteria.identity_verification:
                identity_result = await self._validate_identity_factor(request)
                if identity_result.is_right():
                    factor_results[TrustFactor.IDENTITY] = identity_result.get_right()
            
            # Validate device factor
            if request.criteria.device_compliance:
                device_result = await self._validate_device_factor(request)
                if device_result.is_right():
                    factor_results[TrustFactor.DEVICE] = device_result.get_right()
            
            # Validate location factor
            if request.criteria.location_verification:
                location_result = await self._validate_location_factor(request)
                if location_result.is_right():
                    factor_results[TrustFactor.LOCATION] = location_result.get_right()
            
            # Validate behavior factor
            if request.criteria.behavior_analysis:
                behavior_result = await self._validate_behavior_factor(request)
                if behavior_result.is_right():
                    factor_results[TrustFactor.BEHAVIOR] = behavior_result.get_right()
            
            # Validate network factor
            if request.criteria.network_security:
                network_result = await self._validate_network_factor(request)
                if network_result.is_right():
                    factor_results[TrustFactor.NETWORK] = network_result.get_right()
            
            # Validate temporal factor
            if request.criteria.temporal_validation:
                temporal_result = await self._validate_temporal_factor(request)
                if temporal_result.is_right():
                    factor_results[TrustFactor.TEMPORAL] = temporal_result.get_right()
            
            return Either.right(factor_results)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Trust factor validation failed: {str(e)}",
                "FACTOR_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_identity_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate identity trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Identity validation logic
            identity_score = 0.8  # Placeholder - would integrate with identity providers
            confidence = 0.9
            evidence = {
                "authentication_method": "strong",
                "identity_provider": "enterprise_ldap",
                "multi_factor": True,
                "last_verification": datetime.now(UTC).isoformat()
            }
            warnings = []
            
            # Check for identity risks
            if request.context.user_id and self._is_high_risk_user(request.context.user_id):
                identity_score *= 0.8
                warnings.append("User identified as high-risk")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.IDENTITY,
                score=create_trust_score(identity_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Identity validation failed: {str(e)}",
                "IDENTITY_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_device_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate device trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Device validation logic
            device_score = 0.75  # Placeholder - would check device compliance
            confidence = 0.85
            evidence = {
                "device_id": request.context.device_id,
                "compliance_status": "compliant",
                "security_patch_level": "current",
                "encryption_enabled": True,
                "last_scan": datetime.now(UTC).isoformat()
            }
            warnings = []
            
            # Check device risks
            if not request.context.device_id:
                device_score *= 0.5
                warnings.append("No device identifier provided")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.DEVICE,
                score=create_trust_score(device_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Device validation failed: {str(e)}",
                "DEVICE_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_location_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate location trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Location validation logic
            location_score = 0.7  # Placeholder - would check IP geolocation
            confidence = 0.8
            evidence = {
                "ip_address": request.context.ip_address,
                "location": request.context.location,
                "geolocation_confidence": "high",
                "vpn_detected": False,
                "known_location": True
            }
            warnings = []
            
            # Check location risks
            if self._is_high_risk_location(request.context.location):
                location_score *= 0.6
                warnings.append("Access from high-risk location")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.LOCATION,
                score=create_trust_score(location_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Location validation failed: {str(e)}",
                "LOCATION_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_behavior_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate behavior trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Behavior validation logic
            behavior_score = 0.85  # Placeholder - would analyze user behavior patterns
            confidence = 0.75
            evidence = {
                "behavior_pattern": "normal",
                "anomaly_score": 0.1,
                "typical_access_time": True,
                "expected_resources": True,
                "user_agent_consistent": True
            }
            warnings = []
            
            # Check behavioral anomalies
            if self._detect_behavioral_anomaly(request):
                behavior_score *= 0.7
                warnings.append("Behavioral anomaly detected")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.BEHAVIOR,
                score=create_trust_score(behavior_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Behavior validation failed: {str(e)}",
                "BEHAVIOR_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_network_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate network trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Network validation logic
            network_score = 0.8  # Placeholder - would check network security
            confidence = 0.9
            evidence = {
                "network_segment": "trusted",
                "encryption_in_transit": True,
                "firewall_rules": "applied",
                "intrusion_detection": "active",
                "network_reputation": "good"
            }
            warnings = []
            
            # Check network risks
            if self._is_untrusted_network(request.context.ip_address):
                network_score *= 0.5
                warnings.append("Access from untrusted network")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.NETWORK,
                score=create_trust_score(network_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Network validation failed: {str(e)}",
                "NETWORK_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    async def _validate_temporal_factor(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, TrustFactorResult]:
        """Validate temporal trust factor."""
        try:
            start_time = datetime.now(UTC)
            
            # Temporal validation logic
            temporal_score = 0.9  # Placeholder - would check time-based factors
            confidence = 0.95
            evidence = {
                "access_time": "business_hours",
                "session_duration": "normal",
                "time_since_last_auth": "recent",
                "timezone_consistent": True,
                "temporal_pattern": "expected"
            }
            warnings = []
            
            # Check temporal risks
            current_hour = datetime.now(UTC).hour
            if current_hour < 6 or current_hour > 22:  # Outside business hours
                temporal_score *= 0.8
                warnings.append("Access outside normal business hours")
            
            validation_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            result = TrustFactorResult(
                factor=TrustFactor.TEMPORAL,
                score=create_trust_score(temporal_score),
                confidence=confidence,
                evidence=evidence,
                warnings=warnings,
                validation_time=validation_time,
                status=ValidationStatus.COMPLETED
            )
            
            return Either.right(result)
            
        except Exception as e:
            return Either.left(TrustValidationError(
                f"Temporal validation failed: {str(e)}",
                "TEMPORAL_VALIDATION_ERROR",
                SecurityOperation.VALIDATE
            ))
    
    def _calculate_composite_trust_score(
        self, 
        factors: Dict[TrustFactor, TrustFactorResult]
    ) -> TrustScore:
        """Calculate composite trust score from factor results."""
        if not factors:
            return create_trust_score(0.0)
        
        # Weight factors by importance and confidence
        weighted_scores = []
        total_weight = 0.0
        
        factor_weights = {
            TrustFactor.IDENTITY: 0.3,
            TrustFactor.DEVICE: 0.25,
            TrustFactor.BEHAVIOR: 0.2,
            TrustFactor.LOCATION: 0.15,
            TrustFactor.NETWORK: 0.1,
            TrustFactor.TEMPORAL: 0.05,
            TrustFactor.CREDENTIAL: 0.2,
            TrustFactor.REPUTATION: 0.1
        }
        
        for factor, result in factors.items():
            if result.status == ValidationStatus.COMPLETED:
                base_weight = factor_weights.get(factor, 0.1)
                confidence_weight = base_weight * result.confidence
                weighted_scores.append(float(result.score) * confidence_weight)
                total_weight += confidence_weight
        
        if total_weight == 0:
            return create_trust_score(0.0)
        
        composite_score = sum(weighted_scores) / total_weight
        return create_trust_score(min(1.0, max(0.0, composite_score)))
    
    def _assess_risk_factors(
        self, 
        request: TrustValidationRequest, 
        factors: Dict[TrustFactor, TrustFactorResult]
    ) -> List[str]:
        """Assess risk factors from validation results."""
        risk_factors = []
        
        for factor, result in factors.items():
            # Add warnings as risk factors
            risk_factors.extend(result.warnings)
            
            # Check for low scores
            if result.score < 0.5:
                risk_factors.append(f"Low {factor.value} trust score ({result.score:.2f})")
            
            # Check for low confidence
            if result.confidence < 0.7:
                risk_factors.append(f"Low confidence in {factor.value} validation ({result.confidence:.2f})")
        
        # Check context-specific risks
        if request.context.trust_level == TrustLevel.UNKNOWN:
            risk_factors.append("Unknown trust level")
        
        if request.context.risk_score > 0.7:
            risk_factors.append(f"High context risk score ({request.context.risk_score:.2f})")
        
        return list(set(risk_factors))  # Remove duplicates
    
    def _generate_trust_recommendations(
        self, 
        factors: Dict[TrustFactor, TrustFactorResult], 
        trust_level: TrustLevel
    ) -> List[str]:
        """Generate trust improvement recommendations."""
        recommendations = []
        
        # Level-based recommendations
        if trust_level == TrustLevel.LOW:
            recommendations.append("Consider additional authentication factors")
            recommendations.append("Review device compliance status")
        elif trust_level == TrustLevel.MEDIUM:
            recommendations.append("Monitor for behavioral anomalies")
            recommendations.append("Validate device security posture")
        elif trust_level == TrustLevel.HIGH:
            recommendations.append("Maintain current security posture")
        
        # Factor-specific recommendations
        for factor, result in factors.items():
            if result.score < 0.6:
                if factor == TrustFactor.IDENTITY:
                    recommendations.append("Strengthen identity verification")
                elif factor == TrustFactor.DEVICE:
                    recommendations.append("Update device security configuration")
                elif factor == TrustFactor.LOCATION:
                    recommendations.append("Verify location and network security")
                elif factor == TrustFactor.BEHAVIOR:
                    recommendations.append("Review recent user activity patterns")
                elif factor == TrustFactor.NETWORK:
                    recommendations.append("Enhance network security controls")
        
        return list(set(recommendations))  # Remove duplicates
    
    def _calculate_validation_expiration(
        self, 
        trust_level: TrustLevel, 
        criteria: ValidationCriteria
    ) -> Optional[datetime]:
        """Calculate when validation expires based on trust level."""
        # Higher trust = longer validity
        base_duration = {
            TrustLevel.UNTRUSTED: 0,      # Immediate expiration
            TrustLevel.LOW: 300,          # 5 minutes
            TrustLevel.MEDIUM: 1800,      # 30 minutes
            TrustLevel.HIGH: 3600,        # 1 hour
            TrustLevel.VERIFIED: 7200,    # 2 hours
            TrustLevel.UNKNOWN: 600       # 10 minutes
        }
        
        duration_seconds = base_duration.get(trust_level, 600)
        
        if duration_seconds == 0:
            return datetime.now(UTC)  # Already expired
        
        return datetime.now(UTC) + timedelta(seconds=duration_seconds)
    
    def _validate_request(
        self, 
        request: TrustValidationRequest
    ) -> Either[TrustValidationError, None]:
        """Validate trust validation request."""
        # Validate security context
        context_validation = validate_security_context(request.context)
        if context_validation.is_left():
            return Either.left(TrustValidationError(
                f"Invalid security context: {context_validation.get_left().message}",
                "INVALID_CONTEXT",
                SecurityOperation.VALIDATE
            ))
        
        # Check for duplicate active validation
        for active_request in self.active_validations.values():
            if (active_request.target_id == request.target_id and 
                active_request.scope == request.scope):
                return Either.left(TrustValidationError(
                    f"Validation already in progress for {request.target_id}",
                    "DUPLICATE_VALIDATION",
                    SecurityOperation.VALIDATE
                ))
        
        return Either.right(None)
    
    def _get_cached_trust_score(
        self, 
        target_id: str, 
        scope: ValidationScope
    ) -> Optional[Tuple[TrustScore, datetime]]:
        """Get cached trust score if available and valid."""
        cache_key = f"{target_id}:{scope.value}"
        return self.trust_cache.get(cache_key)
    
    def _create_cached_validation_result(
        self, 
        request: TrustValidationRequest, 
        trust_score: TrustScore, 
        cached_time: datetime
    ) -> Either[TrustValidationError, TrustValidationResult]:
        """Create validation result from cached data."""
        trust_level = determine_trust_level(trust_score)
        
        cached_result = TrustValidationResult(
            validation_id=request.validation_id,
            scope=request.scope,
            target_id=request.target_id,
            trust_score=trust_score,
            trust_level=trust_level,
            validation_timestamp=cached_time,
            criteria_results={"cached": True},
            risk_factors=[],
            recommendations=["Using cached trust score"],
            expires_at=cached_time + timedelta(minutes=10),
            metadata={
                "cache_hit": True,
                "cached_at": cached_time.isoformat(),
                "validation_duration_ms": 1.0
            }
        )
        
        return Either.right(cached_result)
    
    def _update_validation_metrics(self, validation_duration: float) -> None:
        """Update validation performance metrics."""
        self.validation_count += 1
        
        # Update average validation time
        if self.validation_count == 1:
            self.average_validation_time = validation_duration
        else:
            alpha = 0.1  # Exponential moving average factor
            self.average_validation_time = (
                alpha * validation_duration + 
                (1 - alpha) * self.average_validation_time
            )
    
    async def _run_continuous_validation(
        self, 
        target_id: str, 
        config: ContinuousValidationConfig
    ) -> None:
        """Run continuous validation for a target."""
        try:
            while target_id in self.continuous_configs:
                current_config = self.continuous_configs[target_id]
                if not current_config.enabled:
                    break
                
                # Perform validation for each monitored factor
                for scope in [ValidationScope.USER, ValidationScope.DEVICE, ValidationScope.SESSION]:
                    try:
                        # Create basic validation request
                        validation_id = create_validation_id()
                        context = SecurityContext(
                            context_id=f"continuous_{target_id}",
                            user_id=target_id
                        )
                        criteria = ValidationCriteria()
                        
                        request = TrustValidationRequest(
                            validation_id=validation_id,
                            scope=scope,
                            target_id=target_id,
                            context=context,
                            criteria=criteria,
                            priority="normal"
                        )
                        
                        # Perform validation
                        result = await self.validate_trust(request)
                        
                        # Check if action needed
                        if result.is_right():
                            validation_result = result.get_right()
                            if validation_result.trust_score < config.score_threshold:
                                # Trigger alerts or remediation
                                await self._handle_low_trust_score(
                                    target_id, validation_result, config
                                )
                    
                    except Exception as e:
                        # Log error but continue monitoring
                        pass
                
                # Wait for next validation cycle
                await asyncio.sleep(current_config.validation_interval)
                
        except Exception as e:
            # Remove from continuous monitoring on error
            if target_id in self.continuous_configs:
                del self.continuous_configs[target_id]
    
    async def _handle_low_trust_score(
        self, 
        target_id: str, 
        result: TrustValidationResult, 
        config: ContinuousValidationConfig
    ) -> None:
        """Handle low trust score in continuous monitoring."""
        try:
            # Add to threat indicators
            if target_id not in self.threat_indicators:
                self.threat_indicators[target_id] = []
            
            self.threat_indicators[target_id].append(
                f"Low trust score: {result.trust_score:.2f} at {result.validation_timestamp}"
            )
            
            # Auto-remediation if enabled
            if config.auto_remediation:
                # Implement remediation actions
                pass
            
        except Exception as e:
            # Log but don't fail
            pass
    
    # Helper methods for risk assessment
    def _is_high_risk_user(self, user_id: str) -> bool:
        """Check if user is considered high risk."""
        # Placeholder - would check user risk database
        return False
    
    def _is_high_risk_location(self, location: Optional[str]) -> bool:
        """Check if location is considered high risk."""
        # Placeholder - would check location risk database
        return False
    
    def _is_untrusted_network(self, ip_address: Optional[str]) -> bool:
        """Check if network is untrusted."""
        # Placeholder - would check network reputation
        return False
    
    def _detect_behavioral_anomaly(self, request: TrustValidationRequest) -> bool:
        """Detect behavioral anomalies."""
        # Placeholder - would use ML models for anomaly detection
        return False


# Utility functions for trust validation
def create_trust_validation_request(
    target_id: str,
    scope: ValidationScope,
    context: SecurityContext,
    criteria: Optional[ValidationCriteria] = None,
    priority: str = "normal"
) -> TrustValidationRequest:
    """Create a trust validation request with default values."""
    if criteria is None:
        criteria = ValidationCriteria()
    
    return TrustValidationRequest(
        validation_id=create_validation_id(),
        scope=scope,
        target_id=target_id,
        context=context,
        criteria=criteria,
        priority=priority
    )


def create_continuous_validation_config(
    enabled: bool = True,
    validation_interval: int = 300,
    score_threshold: float = 0.7,
    auto_remediation: bool = False
) -> ContinuousValidationConfig:
    """Create continuous validation configuration with sensible defaults."""
    return ContinuousValidationConfig(
        enabled=enabled,
        validation_interval=validation_interval,
        score_threshold=score_threshold,
        auto_remediation=auto_remediation
    )
