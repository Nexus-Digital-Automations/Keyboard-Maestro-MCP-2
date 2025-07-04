"""
Biometric Authentication Manager - TASK_67 Phase 2 Core Biometric Engine

Multi-modal biometric authentication system with enterprise-grade security, privacy protection,
liveness detection, and anti-spoofing capabilities.

Architecture: Multi-Modal Authentication + Design by Contract + Type Safety + Privacy Protection
Performance: <100ms authentication, <50ms liveness detection, <200ms multi-factor verification
Security: Encrypted templates, liveness detection, anti-spoofing, secure session management
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
import asyncio
import logging
import hashlib
import secrets
import json
from pathlib import Path

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.biometric_architecture import (
    BiometricModality, SecurityLevel, PrivacyLevel, BiometricError,
    BiometricTemplate, AuthenticationRequest, AuthenticationResult,
    UserProfile, BiometricSession, BiometricConfiguration,
    BiometricTemplateId, BiometricSessionId, UserProfileId, BiometricHash,
    BiometricData, ConfidenceScores, SecurityMetrics,
    generate_template_id, generate_session_id, generate_profile_id,
    calculate_feature_hash, calculate_security_score, create_default_biometric_config
)

logger = logging.getLogger(__name__)


class BiometricAuthenticationManager:
    """Multi-modal biometric authentication system with privacy protection."""
    
    def __init__(self, config: Optional[BiometricConfiguration] = None):
        self.config = config or create_default_biometric_config()
        self.templates: Dict[BiometricTemplateId, BiometricTemplate] = {}
        self.user_profiles: Dict[UserProfileId, UserProfile] = {}
        self.active_sessions: Dict[BiometricSessionId, BiometricSession] = {}
        self.authentication_cache: Dict[str, AuthenticationResult] = {}
        self.security_metrics: SecurityMetrics = {
            "total_attempts": 0.0,
            "successful_authentications": 0.0,
            "failed_attempts": 0.0,
            "liveness_failures": 0.0,
            "spoofing_attempts": 0.0
        }
    
    @require(lambda self, request: isinstance(request, AuthenticationRequest))
    @ensure(lambda result: result.is_success() or result.is_error())
    async def authenticate_user(self, request: AuthenticationRequest, 
                              biometric_data: BiometricData) -> Either[BiometricError, AuthenticationResult]:
        """Perform multi-modal biometric authentication with security validation."""
        try:
            start_time = datetime.now(UTC)
            self.security_metrics["total_attempts"] += 1
            
            # Validate authentication request
            validation_result = self._validate_authentication_request(request)
            if validation_result.is_error():
                return validation_result
            
            # Perform liveness detection if required
            if request.liveness_required:
                liveness_result = await self._perform_liveness_detection(
                    request.modalities, biometric_data
                )
                if liveness_result.is_error():
                    self.security_metrics["liveness_failures"] += 1
                    return liveness_result
                
                liveness_verified = liveness_result.value
            else:
                liveness_verified = False
            
            # Perform anti-spoofing detection
            anti_spoofing_result = await self._perform_anti_spoofing_detection(
                request.modalities, biometric_data
            )
            if anti_spoofing_result.is_error():
                self.security_metrics["spoofing_attempts"] += 1
                return anti_spoofing_result
            
            anti_spoofing_passed = anti_spoofing_result.value
            
            # Perform biometric matching
            matching_result = await self._perform_biometric_matching(
                request.modalities, biometric_data, request.user_context
            )
            if matching_result.is_error():
                self.security_metrics["failed_attempts"] += 1
                return matching_result
            
            user_profile_id, confidence_scores = matching_result.value
            
            # Calculate security score
            security_score = calculate_security_score(
                confidence_scores, liveness_verified, anti_spoofing_passed
            )
            
            # Validate security requirements
            if not self._meets_security_requirements(
                security_score, confidence_scores, request.security_level
            ):
                return Either.error(BiometricError.authentication_failed("insufficient_security"))
            
            # Calculate processing time
            processing_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            
            # Create authentication result
            result = AuthenticationResult(
                session_id=request.session_id,
                success=True,
                user_profile_id=user_profile_id,
                authenticated_modalities=list(confidence_scores.keys()),
                confidence_scores=confidence_scores,
                security_score=security_score,
                liveness_verified=liveness_verified,
                processing_time_ms=processing_time,
                authenticated_at=datetime.now(UTC),
                valid_until=datetime.now(UTC) + timedelta(hours=self.config.session_timeout_hours),
                security_warnings=self._generate_security_warnings(
                    security_score, confidence_scores, liveness_verified, anti_spoofing_passed
                )
            )
            
            # Create or update biometric session
            session_result = await self._create_biometric_session(request, result)
            if session_result.is_error():
                return session_result
            
            # Update user profile last authenticated time
            if user_profile_id in self.user_profiles:
                await self._update_user_last_authenticated(user_profile_id)
            
            # Update security metrics
            self.security_metrics["successful_authentications"] += 1
            
            # Cache authentication result
            self._cache_authentication_result(result)
            
            logger.info(f"Biometric authentication successful for user {user_profile_id} "
                       f"with security score {security_score:.3f}")
            
            return Either.success(result)
            
        except Exception as e:
            self.security_metrics["failed_attempts"] += 1
            logger.error(f"Biometric authentication failed: {e}")
            return Either.error(BiometricError(f"Authentication failed: {str(e)}"))
    
    @require(lambda self, user_identity: len(user_identity) > 0)
    @require(lambda self, modalities: len(modalities) > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def enroll_user(self, user_identity: str, modalities: List[BiometricModality],
                         biometric_data: BiometricData,
                         personalization_preferences: Optional[Dict[str, Any]] = None) -> Either[BiometricError, UserProfile]:
        """Enroll new user with biometric templates and personalization settings."""
        try:
            # Validate enrollment data
            if not all(modality in biometric_data for modality in modalities):
                return Either.error(BiometricError("Missing biometric data for enrollment"))
            
            # Check if user already exists
            profile_id = generate_profile_id(user_identity)
            if profile_id in self.user_profiles:
                return Either.error(BiometricError(f"User already enrolled: {user_identity}"))
            
            # Process biometric templates
            templates: Dict[BiometricModality, BiometricTemplateId] = {}
            enrolled_modalities: Set[BiometricModality] = set()
            
            for modality in modalities:
                template_result = await self._create_biometric_template(
                    modality, biometric_data[modality]
                )
                if template_result.is_error():
                    return template_result
                
                template = template_result.value
                templates[modality] = template.template_id
                enrolled_modalities.add(modality)
                
                # Store template
                self.templates[template.template_id] = template
            
            # Create user profile
            user_profile = UserProfile(
                profile_id=profile_id,
                user_identity=user_identity,
                enrolled_modalities=enrolled_modalities,
                biometric_templates=templates,
                personalization_preferences=personalization_preferences or {},
                accessibility_settings={},
                behavioral_patterns={},
                privacy_settings={"privacy_level": self.config.privacy_level.value},
                created_at=datetime.now(UTC),
                last_updated=datetime.now(UTC)
            )
            
            # Store user profile
            self.user_profiles[profile_id] = user_profile
            
            logger.info(f"User enrolled successfully: {user_identity} with modalities {modalities}")
            
            return Either.success(user_profile)
            
        except Exception as e:
            logger.error(f"User enrollment failed: {e}")
            return Either.error(BiometricError(f"Enrollment failed: {str(e)}"))
    
    @require(lambda self, session_id: len(session_id) > 0)
    async def get_session(self, session_id: BiometricSessionId) -> Either[BiometricError, BiometricSession]:
        """Get active biometric session by ID."""
        try:
            if session_id not in self.active_sessions:
                return Either.error(BiometricError(f"Session not found: {session_id}"))
            
            session = self.active_sessions[session_id]
            
            # Check if session is expired
            if session.is_expired():
                await self._cleanup_expired_session(session_id)
                return Either.error(BiometricError(f"Session expired: {session_id}"))
            
            return Either.success(session)
            
        except Exception as e:
            logger.error(f"Failed to get session: {e}")
            return Either.error(BiometricError(f"Session retrieval failed: {str(e)}"))
    
    async def verify_continuous_authentication(self, session_id: BiometricSessionId,
                                             biometric_data: BiometricData) -> Either[BiometricError, bool]:
        """Verify continuous authentication for active session."""
        try:
            session_result = await self.get_session(session_id)
            if session_result.is_error():
                return session_result
            
            session = session_result.value
            
            if not session.continuous_monitoring:
                return Either.success(True)  # Not required for this session
            
            # Perform quick verification
            if not session.user_profile_id:
                return Either.error(BiometricError("No user profile for continuous verification"))
            
            user_profile = self.user_profiles.get(session.user_profile_id)
            if not user_profile:
                return Either.error(BiometricError("User profile not found"))
            
            # Quick confidence check with lower threshold
            verification_result = await self._quick_biometric_verification(
                user_profile, biometric_data
            )
            
            if verification_result.is_success():
                # Update session activity
                updated_session = session.update_activity()
                self.active_sessions[session_id] = updated_session
            
            return verification_result
            
        except Exception as e:
            logger.error(f"Continuous authentication verification failed: {e}")
            return Either.error(BiometricError(f"Continuous verification failed: {str(e)}"))
    
    async def get_security_metrics(self) -> SecurityMetrics:
        """Get biometric system security metrics."""
        # Calculate success rate
        total = self.security_metrics["total_attempts"]
        if total > 0:
            success_rate = self.security_metrics["successful_authentications"] / total
            failure_rate = self.security_metrics["failed_attempts"] / total
            liveness_failure_rate = self.security_metrics["liveness_failures"] / total
            spoofing_rate = self.security_metrics["spoofing_attempts"] / total
        else:
            success_rate = failure_rate = liveness_failure_rate = spoofing_rate = 0.0
        
        return {
            **self.security_metrics,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "liveness_failure_rate": liveness_failure_rate,
            "spoofing_rate": spoofing_rate,
            "active_sessions": float(len(self.active_sessions)),
            "enrolled_users": float(len(self.user_profiles)),
            "total_templates": float(len(self.templates))
        }
    
    # Private helper methods
    
    def _validate_authentication_request(self, request: AuthenticationRequest) -> Either[BiometricError, None]:
        """Validate authentication request parameters."""
        try:
            # Check modalities are enabled
            for modality in request.modalities:
                if not self.config.is_modality_enabled(modality):
                    return Either.error(BiometricError(f"Modality not enabled: {modality.value}"))
            
            # Validate multi-factor requirements
            if request.multi_factor and len(request.modalities) < 2:
                return Either.error(BiometricError("Multi-factor requires at least 2 modalities"))
            
            # Check timeout
            if request.timeout_seconds <= 0 or request.timeout_seconds > 300:
                return Either.error(BiometricError("Invalid timeout value"))
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(BiometricError(f"Request validation failed: {str(e)}"))
    
    async def _perform_liveness_detection(self, modalities: List[BiometricModality],
                                        biometric_data: BiometricData) -> Either[BiometricError, bool]:
        """Perform liveness detection for biometric data."""
        try:
            # Implement liveness detection for each modality
            liveness_results = {}
            
            for modality in modalities:
                if modality not in biometric_data:
                    continue
                
                if modality == BiometricModality.FACE:
                    # Face liveness detection (blink, head movement, etc.)
                    liveness_results[modality] = await self._face_liveness_detection(
                        biometric_data[modality]
                    )
                elif modality == BiometricModality.FINGERPRINT:
                    # Fingerprint liveness detection (pulse, temperature, etc.)
                    liveness_results[modality] = await self._fingerprint_liveness_detection(
                        biometric_data[modality]
                    )
                elif modality == BiometricModality.VOICE:
                    # Voice liveness detection (challenge-response, etc.)
                    liveness_results[modality] = await self._voice_liveness_detection(
                        biometric_data[modality]
                    )
                else:
                    # Default liveness check
                    liveness_results[modality] = True
            
            # All modalities must pass liveness detection
            overall_liveness = all(liveness_results.values()) if liveness_results else False
            
            return Either.success(overall_liveness)
            
        except Exception as e:
            return Either.error(BiometricError.liveness_detection_failed(f"general: {str(e)}"))
    
    async def _perform_anti_spoofing_detection(self, modalities: List[BiometricModality],
                                             biometric_data: BiometricData) -> Either[BiometricError, bool]:
        """Perform anti-spoofing detection for biometric data."""
        try:
            # Implement anti-spoofing for each modality
            spoofing_results = {}
            
            for modality in modalities:
                if modality not in biometric_data:
                    continue
                
                if modality == BiometricModality.FACE:
                    # Face anti-spoofing (3D analysis, texture analysis, etc.)
                    spoofing_results[modality] = await self._face_anti_spoofing(
                        biometric_data[modality]
                    )
                elif modality == BiometricModality.FINGERPRINT:
                    # Fingerprint anti-spoofing (ridge analysis, etc.)
                    spoofing_results[modality] = await self._fingerprint_anti_spoofing(
                        biometric_data[modality]
                    )
                else:
                    # Default anti-spoofing check
                    spoofing_results[modality] = True
            
            # All modalities must pass anti-spoofing
            overall_anti_spoofing = all(spoofing_results.values()) if spoofing_results else False
            
            return Either.success(overall_anti_spoofing)
            
        except Exception as e:
            return Either.error(BiometricError(f"Anti-spoofing detection failed: {str(e)}"))
    
    async def _perform_biometric_matching(self, modalities: List[BiometricModality],
                                        biometric_data: BiometricData,
                                        user_context: Optional[str]) -> Either[BiometricError, Tuple[UserProfileId, ConfidenceScores]]:
        """Perform biometric template matching."""
        try:
            confidence_scores: ConfidenceScores = {}
            best_match_profile: Optional[UserProfileId] = None
            best_overall_score = 0.0
            
            # If user context provided, try to match against specific user first
            if user_context:
                context_result = await self._match_against_user_context(
                    user_context, modalities, biometric_data
                )
                if context_result.is_success():
                    profile_id, scores = context_result.value
                    if self._meets_confidence_threshold(scores):
                        return Either.success((profile_id, scores))
            
            # Perform matching against all enrolled users
            for profile_id, user_profile in self.user_profiles.items():
                if not user_profile.is_active:
                    continue
                
                # Check if user has any of the required modalities
                available_modalities = set(modalities) & user_profile.enrolled_modalities
                if not available_modalities:
                    continue
                
                # Match against available modalities
                user_scores: ConfidenceScores = {}
                
                for modality in available_modalities:
                    if modality not in biometric_data:
                        continue
                    
                    template_id = user_profile.get_template_id(modality)
                    if not template_id or template_id not in self.templates:
                        continue
                    
                    template = self.templates[template_id]
                    if template.is_expired():
                        continue
                    
                    # Perform template matching
                    match_score = await self._match_biometric_template(
                        template, biometric_data[modality]
                    )
                    
                    if match_score >= self.config.confidence_threshold:
                        user_scores[modality] = match_score
                
                # Calculate overall score for this user
                if user_scores:
                    overall_score = sum(user_scores.values()) / len(user_scores)
                    
                    # Apply multi-factor bonus
                    if len(user_scores) > 1:
                        overall_score *= 1.1
                    
                    if overall_score > best_overall_score:
                        best_overall_score = overall_score
                        best_match_profile = profile_id
                        confidence_scores = user_scores
            
            if best_match_profile and self._meets_confidence_threshold(confidence_scores):
                return Either.success((best_match_profile, confidence_scores))
            else:
                return Either.error(BiometricError.authentication_failed("no_match"))
                
        except Exception as e:
            return Either.error(BiometricError(f"Biometric matching failed: {str(e)}"))
    
    async def _create_biometric_template(self, modality: BiometricModality,
                                       biometric_data: bytes) -> Either[BiometricError, BiometricTemplate]:
        """Create encrypted biometric template from raw data."""
        try:
            # Extract features from biometric data
            features_result = await self._extract_biometric_features(modality, biometric_data)
            if features_result.is_error():
                return features_result
            
            features, quality_score = features_result.value
            
            # Check quality threshold
            if quality_score < self.config.quality_threshold:
                return Either.error(BiometricError(f"Biometric quality too low: {quality_score}"))
            
            # Encrypt features
            encrypted_features = await self._encrypt_biometric_features(features)
            
            # Generate template
            template = BiometricTemplate(
                template_id=generate_template_id(),
                modality=modality,
                encrypted_data=encrypted_features,
                feature_hash=calculate_feature_hash(features),
                quality_score=quality_score,
                created_at=datetime.now(UTC),
                expires_at=datetime.now(UTC) + timedelta(days=365) if self.config.compliance_mode else None
            )
            
            return Either.success(template)
            
        except Exception as e:
            return Either.error(BiometricError(f"Template creation failed: {str(e)}"))
    
    async def _create_biometric_session(self, request: AuthenticationRequest,
                                      result: AuthenticationResult) -> Either[BiometricError, BiometricSession]:
        """Create new biometric session."""
        try:
            session = BiometricSession(
                session_id=request.session_id,
                user_profile_id=result.user_profile_id,
                authentication_result=result,
                continuous_monitoring=request.continuous_auth,
                security_level=request.security_level,
                created_at=datetime.now(UTC),
                expires_at=result.valid_until or datetime.now(UTC) + timedelta(hours=8),
                last_activity=datetime.now(UTC)
            )
            
            self.active_sessions[request.session_id] = session
            
            return Either.success(session)
            
        except Exception as e:
            return Either.error(BiometricError(f"Session creation failed: {str(e)}"))
    
    def _meets_security_requirements(self, security_score: float,
                                   confidence_scores: ConfidenceScores,
                                   security_level: SecurityLevel) -> bool:
        """Check if authentication meets security requirements."""
        if security_level == SecurityLevel.LOW:
            return security_score >= 0.6
        elif security_level == SecurityLevel.MEDIUM:
            return security_score >= 0.8
        elif security_level == SecurityLevel.HIGH:
            return security_score >= 0.9 and len(confidence_scores) >= 1
        elif security_level == SecurityLevel.CRITICAL:
            return security_score >= 0.95 and len(confidence_scores) >= 2
        return False
    
    def _meets_confidence_threshold(self, confidence_scores: ConfidenceScores) -> bool:
        """Check if confidence scores meet threshold."""
        if not confidence_scores:
            return False
        return all(score >= self.config.confidence_threshold for score in confidence_scores.values())
    
    def _generate_security_warnings(self, security_score: float,
                                  confidence_scores: ConfidenceScores,
                                  liveness_verified: bool,
                                  anti_spoofing_passed: bool) -> List[str]:
        """Generate security warnings for authentication result."""
        warnings = []
        
        if security_score < 0.9:
            warnings.append("Security score below recommended threshold")
        
        if not liveness_verified:
            warnings.append("Liveness detection not performed or failed")
        
        if not anti_spoofing_passed:
            warnings.append("Anti-spoofing detection failed")
        
        if len(confidence_scores) == 1:
            warnings.append("Single-factor authentication used")
        
        low_confidence_modalities = [
            modality.value for modality, score in confidence_scores.items()
            if score < 0.9
        ]
        if low_confidence_modalities:
            warnings.append(f"Low confidence for modalities: {', '.join(low_confidence_modalities)}")
        
        return warnings
    
    # Placeholder methods for actual biometric processing
    # These would integrate with actual biometric libraries/SDKs
    
    async def _face_liveness_detection(self, face_data: bytes) -> bool:
        """Perform face liveness detection."""
        # Placeholder implementation
        await asyncio.sleep(0.01)  # Simulate processing
        return len(face_data) > 1024  # Simple quality check
    
    async def _fingerprint_liveness_detection(self, fingerprint_data: bytes) -> bool:
        """Perform fingerprint liveness detection."""
        await asyncio.sleep(0.01)
        return len(fingerprint_data) > 512
    
    async def _voice_liveness_detection(self, voice_data: bytes) -> bool:
        """Perform voice liveness detection."""
        await asyncio.sleep(0.01)
        return len(voice_data) > 2048
    
    async def _face_anti_spoofing(self, face_data: bytes) -> bool:
        """Perform face anti-spoofing detection."""
        await asyncio.sleep(0.01)
        return True  # Placeholder
    
    async def _fingerprint_anti_spoofing(self, fingerprint_data: bytes) -> bool:
        """Perform fingerprint anti-spoofing detection."""
        await asyncio.sleep(0.01)
        return True
    
    async def _extract_biometric_features(self, modality: BiometricModality,
                                        biometric_data: bytes) -> Either[BiometricError, Tuple[bytes, float]]:
        """Extract features from biometric data."""
        # Placeholder implementation
        await asyncio.sleep(0.02)
        
        # Simulate feature extraction with quality score
        features = hashlib.sha256(biometric_data).digest()
        quality_score = min(len(biometric_data) / 1024.0, 1.0)  # Simple quality metric
        
        return Either.success((features, quality_score))
    
    async def _encrypt_biometric_features(self, features: bytes) -> bytes:
        """Encrypt biometric features for storage."""
        # Placeholder encryption (would use proper encryption in production)
        key = secrets.token_bytes(32)
        encrypted = bytes(a ^ b for a, b in zip(features, key * (len(features) // 32 + 1)))
        return key + encrypted
    
    async def _match_biometric_template(self, template: BiometricTemplate,
                                      biometric_data: bytes) -> float:
        """Match biometric data against template."""
        # Placeholder matching algorithm
        await asyncio.sleep(0.01)
        
        # Simple hash-based matching (would use proper matching in production)
        data_hash = calculate_feature_hash(biometric_data)
        
        # Simulate matching score based on hash similarity
        if data_hash == template.feature_hash:
            return 1.0
        else:
            # Simulate partial matches
            return max(0.0, 0.95 - abs(hash(data_hash) - hash(template.feature_hash)) / 1e10)
    
    async def _match_against_user_context(self, user_context: str,
                                        modalities: List[BiometricModality],
                                        biometric_data: BiometricData) -> Either[BiometricError, Tuple[UserProfileId, ConfidenceScores]]:
        """Match against specific user context."""
        # Find user by context (could be username, email, etc.)
        for profile_id, user_profile in self.user_profiles.items():
            if user_profile.user_identity == user_context:
                # Perform targeted matching
                return await self._match_user_profile(user_profile, modalities, biometric_data)
        
        return Either.error(BiometricError(f"User context not found: {user_context}"))
    
    async def _match_user_profile(self, user_profile: UserProfile,
                                modalities: List[BiometricModality],
                                biometric_data: BiometricData) -> Either[BiometricError, Tuple[UserProfileId, ConfidenceScores]]:
        """Match against specific user profile."""
        confidence_scores: ConfidenceScores = {}
        
        for modality in modalities:
            if modality in user_profile.enrolled_modalities and modality in biometric_data:
                template_id = user_profile.get_template_id(modality)
                if template_id and template_id in self.templates:
                    template = self.templates[template_id]
                    if not template.is_expired():
                        score = await self._match_biometric_template(
                            template, biometric_data[modality]
                        )
                        if score >= self.config.confidence_threshold:
                            confidence_scores[modality] = score
        
        if confidence_scores:
            return Either.success((user_profile.profile_id, confidence_scores))
        else:
            return Either.error(BiometricError.authentication_failed("profile_no_match"))
    
    async def _quick_biometric_verification(self, user_profile: UserProfile,
                                          biometric_data: BiometricData) -> Either[BiometricError, bool]:
        """Perform quick biometric verification for continuous authentication."""
        # Use lower threshold for continuous verification
        lower_threshold = max(0.6, self.config.confidence_threshold - 0.2)
        
        for modality, data in biometric_data.items():
            if modality in user_profile.enrolled_modalities:
                template_id = user_profile.get_template_id(modality)
                if template_id and template_id in self.templates:
                    template = self.templates[template_id]
                    if not template.is_expired():
                        score = await self._match_biometric_template(template, data)
                        if score >= lower_threshold:
                            return Either.success(True)
        
        return Either.error(BiometricError("Continuous verification failed"))
    
    async def _update_user_last_authenticated(self, user_profile_id: UserProfileId) -> None:
        """Update user profile last authenticated timestamp."""
        if user_profile_id in self.user_profiles:
            user_profile = self.user_profiles[user_profile_id]
            # Create updated profile (immutable)
            updated_profile = UserProfile(
                profile_id=user_profile.profile_id,
                user_identity=user_profile.user_identity,
                enrolled_modalities=user_profile.enrolled_modalities,
                biometric_templates=user_profile.biometric_templates,
                personalization_preferences=user_profile.personalization_preferences,
                accessibility_settings=user_profile.accessibility_settings,
                behavioral_patterns=user_profile.behavioral_patterns,
                privacy_settings=user_profile.privacy_settings,
                created_at=user_profile.created_at,
                last_updated=datetime.now(UTC),
                last_authenticated=datetime.now(UTC),
                is_active=user_profile.is_active
            )
            self.user_profiles[user_profile_id] = updated_profile
    
    def _cache_authentication_result(self, result: AuthenticationResult) -> None:
        """Cache authentication result for performance."""
        cache_key = f"{result.user_profile_id}_{result.session_id}"
        self.authentication_cache[cache_key] = result
        
        # Limit cache size
        if len(self.authentication_cache) > 1000:
            # Remove oldest entries
            oldest_keys = sorted(self.authentication_cache.keys())[:100]
            for key in oldest_keys:
                del self.authentication_cache[key]
    
    async def _cleanup_expired_session(self, session_id: BiometricSessionId) -> None:
        """Clean up expired session."""
        if session_id in self.active_sessions:
            del self.active_sessions[session_id]
            logger.info(f"Cleaned up expired session: {session_id}")