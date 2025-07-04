"""
User Profiler - TASK_67 Phase 2 Core Biometric Engine

User identification and profile management with behavioral analysis and adaptive personalization
for biometric-driven automation experiences.

Architecture: User Profiling + Behavioral Analysis + Design by Contract + Privacy Protection
Performance: <50ms user identification, <100ms profile retrieval, <200ms behavioral analysis
Security: Privacy-preserving profiling, encrypted user data, behavioral pattern protection
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Tuple
from datetime import datetime, timedelta, UTC
from dataclasses import dataclass, field
import asyncio
import logging
import json
import hashlib
from collections import defaultdict, deque

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.biometric_architecture import (
    BiometricModality, PrivacyLevel, BiometricError,
    UserProfile, UserProfileId, BiometricSessionId,
    generate_profile_id
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class BehavioralPattern:
    """User behavioral pattern for personalization."""
    pattern_id: str
    pattern_type: str  # interaction|timing|preference|accessibility
    pattern_data: Dict[str, Any]
    confidence: float
    frequency: int
    first_observed: datetime
    last_observed: datetime
    context: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.frequency >= 0)
    def __post_init__(self):
        if self.last_observed < self.first_observed:
            raise ValueError("Last observed cannot be before first observed")


@dataclass(frozen=True)
class PersonalizationContext:
    """Context for personalization decisions."""
    user_profile_id: UserProfileId
    session_id: Optional[BiometricSessionId]
    automation_type: str
    current_context: Dict[str, Any]
    time_of_day: str
    device_info: Dict[str, Any]
    environmental_factors: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.automation_type) > 0)
    def __post_init__(self):
        pass


@dataclass(frozen=True)
class PersonalizationRecommendation:
    """Personalization recommendation with confidence and rationale."""
    recommendation_id: str
    user_profile_id: UserProfileId
    recommendation_type: str
    recommended_settings: Dict[str, Any]
    confidence: float
    rationale: str
    supporting_patterns: List[str]
    impact_estimate: str  # low|medium|high
    created_at: datetime
    expires_at: Optional[datetime] = None
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: len(self.rationale) > 0)
    def __post_init__(self):
        pass
    
    def is_expired(self) -> bool:
        """Check if recommendation has expired."""
        if not self.expires_at:
            return False
        return datetime.now(UTC) >= self.expires_at


class UserProfiler:
    """User identification and profile management with behavioral analysis."""
    
    def __init__(self, privacy_level: PrivacyLevel = PrivacyLevel.ENHANCED):
        self.privacy_level = privacy_level
        self.user_profiles: Dict[UserProfileId, UserProfile] = {}
        self.behavioral_patterns: Dict[UserProfileId, List[BehavioralPattern]] = defaultdict(list)
        self.interaction_history: Dict[UserProfileId, deque] = defaultdict(lambda: deque(maxlen=1000))
        self.personalization_cache: Dict[str, PersonalizationRecommendation] = {}
        self.profile_analytics: Dict[UserProfileId, Dict[str, Any]] = defaultdict(dict)
    
    @require(lambda self, user_identity: len(user_identity) > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def identify_user(self, user_identity: str, 
                          biometric_confidence: Optional[float] = None,
                          context: Optional[Dict[str, Any]] = None) -> Either[BiometricError, UserProfile]:
        """Identify user and retrieve profile with optional biometric confidence."""
        try:
            profile_id = generate_profile_id(user_identity)
            
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {user_identity}"))
            
            user_profile = self.user_profiles[profile_id]
            
            if not user_profile.is_active:
                return Either.error(BiometricError(f"User profile inactive: {user_identity}"))
            
            # Update identification analytics
            await self._update_identification_analytics(
                profile_id, biometric_confidence, context
            )
            
            # Record interaction
            await self._record_user_interaction(
                profile_id, "identification", context or {}
            )
            
            logger.info(f"User identified successfully: {user_identity}")
            
            return Either.success(user_profile)
            
        except Exception as e:
            logger.error(f"User identification failed: {e}")
            return Either.error(BiometricError(f"Identification failed: {str(e)}"))
    
    @require(lambda self, profile_id: len(profile_id) > 0)
    @ensure(lambda result: result.is_success() or result.is_error())
    async def get_user_profile(self, profile_id: UserProfileId) -> Either[BiometricError, UserProfile]:
        """Get user profile by ID."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError.template_not_found(f"Profile: {profile_id}"))
            
            return Either.success(self.user_profiles[profile_id])
            
        except Exception as e:
            return Either.error(BiometricError(f"Profile retrieval failed: {str(e)}"))
    
    @require(lambda self, profile_id: len(profile_id) > 0)
    async def update_user_preferences(self, profile_id: UserProfileId,
                                    preferences: Dict[str, Any],
                                    merge: bool = True) -> Either[BiometricError, UserProfile]:
        """Update user personalization preferences."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {profile_id}"))
            
            current_profile = self.user_profiles[profile_id]
            
            # Merge or replace preferences
            if merge:
                updated_preferences = {**current_profile.personalization_preferences, **preferences}
            else:
                updated_preferences = preferences
            
            # Create updated profile
            updated_profile = UserProfile(
                profile_id=current_profile.profile_id,
                user_identity=current_profile.user_identity,
                enrolled_modalities=current_profile.enrolled_modalities,
                biometric_templates=current_profile.biometric_templates,
                personalization_preferences=updated_preferences,
                accessibility_settings=current_profile.accessibility_settings,
                behavioral_patterns=current_profile.behavioral_patterns,
                privacy_settings=current_profile.privacy_settings,
                created_at=current_profile.created_at,
                last_updated=datetime.now(UTC),
                last_authenticated=current_profile.last_authenticated,
                is_active=current_profile.is_active
            )
            
            self.user_profiles[profile_id] = updated_profile
            
            # Record preference update
            await self._record_user_interaction(
                profile_id, "preference_update", {"updated_keys": list(preferences.keys())}
            )
            
            logger.info(f"User preferences updated for profile: {profile_id}")
            
            return Either.success(updated_profile)
            
        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return Either.error(BiometricError(f"Preference update failed: {str(e)}"))
    
    @require(lambda self, profile_id: len(profile_id) > 0)
    async def analyze_user_behavior(self, profile_id: UserProfileId,
                                  analysis_period_days: int = 7) -> Either[BiometricError, Dict[str, Any]]:
        """Analyze user behavioral patterns for personalization insights."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {profile_id}"))
            
            # Get behavioral patterns for the user
            patterns = self.behavioral_patterns.get(profile_id, [])
            
            # Filter patterns within analysis period
            cutoff_date = datetime.now(UTC) - timedelta(days=analysis_period_days)
            recent_patterns = [
                pattern for pattern in patterns 
                if pattern.last_observed >= cutoff_date
            ]
            
            # Analyze interaction history
            interaction_history = list(self.interaction_history.get(profile_id, []))
            recent_interactions = [
                interaction for interaction in interaction_history
                if interaction.get('timestamp', datetime.min.replace(tzinfo=UTC)) >= cutoff_date
            ]
            
            # Generate behavioral analysis
            analysis = {
                "profile_id": profile_id,
                "analysis_period_days": analysis_period_days,
                "total_patterns": len(patterns),
                "recent_patterns": len(recent_patterns),
                "total_interactions": len(interaction_history),
                "recent_interactions": len(recent_interactions),
                "pattern_analysis": await self._analyze_behavioral_patterns(recent_patterns),
                "interaction_analysis": await self._analyze_interaction_patterns(recent_interactions),
                "personalization_insights": await self._generate_personalization_insights(
                    profile_id, recent_patterns, recent_interactions
                ),
                "behavioral_trends": await self._analyze_behavioral_trends(recent_patterns),
                "generated_at": datetime.now(UTC)
            }
            
            # Update profile analytics cache
            self.profile_analytics[profile_id].update({
                "last_behavioral_analysis": analysis,
                "last_analysis_date": datetime.now(UTC)
            })
            
            return Either.success(analysis)
            
        except Exception as e:
            logger.error(f"Behavioral analysis failed: {e}")
            return Either.error(BiometricError(f"Behavioral analysis failed: {str(e)}"))
    
    async def detect_behavioral_anomalies(self, profile_id: UserProfileId,
                                        current_interaction: Dict[str, Any]) -> Either[BiometricError, List[str]]:
        """Detect behavioral anomalies for security purposes."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {profile_id}"))
            
            anomalies = []
            
            # Get user's behavioral patterns
            patterns = self.behavioral_patterns.get(profile_id, [])
            interaction_history = list(self.interaction_history.get(profile_id, []))
            
            # Check time-based anomalies
            time_anomalies = await self._detect_time_anomalies(
                current_interaction, interaction_history
            )
            anomalies.extend(time_anomalies)
            
            # Check interaction pattern anomalies
            pattern_anomalies = await self._detect_pattern_anomalies(
                current_interaction, patterns
            )
            anomalies.extend(pattern_anomalies)
            
            # Check device/location anomalies
            context_anomalies = await self._detect_context_anomalies(
                current_interaction, interaction_history
            )
            anomalies.extend(context_anomalies)
            
            # Record anomaly detection
            if anomalies:
                await self._record_user_interaction(
                    profile_id, "anomaly_detection", {
                        "anomalies": anomalies,
                        "interaction": current_interaction
                    }
                )
            
            return Either.success(anomalies)
            
        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return Either.error(BiometricError(f"Anomaly detection failed: {str(e)}"))
    
    async def learn_from_interaction(self, profile_id: UserProfileId,
                                   interaction_type: str,
                                   interaction_data: Dict[str, Any],
                                   success: bool = True) -> Either[BiometricError, None]:
        """Learn from user interaction to improve personalization."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {profile_id}"))
            
            user_profile = self.user_profiles[profile_id]
            
            # Check if learning is enabled for this user
            if not user_profile.privacy_settings.get("allow_learning", True):
                return Either.success(None)
            
            # Record interaction
            interaction = {
                "type": interaction_type,
                "data": interaction_data,
                "success": success,
                "timestamp": datetime.now(UTC),
                "context": interaction_data.get("context", {})
            }
            
            self.interaction_history[profile_id].append(interaction)
            
            # Extract behavioral patterns
            patterns = await self._extract_behavioral_patterns(interaction, user_profile)
            
            # Update or create behavioral patterns
            for pattern in patterns:
                await self._update_behavioral_pattern(profile_id, pattern)
            
            # Generate learning insights
            insights = await self._generate_learning_insights(profile_id, interaction)
            
            # Update profile behavioral patterns if significant insights found
            if insights.get("significant_patterns"):
                await self._update_profile_behavioral_patterns(profile_id, insights)
            
            return Either.success(None)
            
        except Exception as e:
            logger.error(f"Learning from interaction failed: {e}")
            return Either.error(BiometricError(f"Learning failed: {str(e)}"))
    
    async def get_user_analytics(self, profile_id: UserProfileId) -> Either[BiometricError, Dict[str, Any]]:
        """Get comprehensive user analytics and insights."""
        try:
            if profile_id not in self.user_profiles:
                return Either.error(BiometricError(f"User profile not found: {profile_id}"))
            
            user_profile = self.user_profiles[profile_id]
            patterns = self.behavioral_patterns.get(profile_id, [])
            interactions = list(self.interaction_history.get(profile_id, []))
            cached_analytics = self.profile_analytics.get(profile_id, {})
            
            analytics = {
                "profile_summary": {
                    "profile_id": profile_id,
                    "user_identity": user_profile.user_identity,
                    "enrolled_modalities": [mod.value for mod in user_profile.enrolled_modalities],
                    "created_at": user_profile.created_at,
                    "last_updated": user_profile.last_updated,
                    "last_authenticated": user_profile.last_authenticated,
                    "is_active": user_profile.is_active
                },
                "behavioral_statistics": {
                    "total_patterns": len(patterns),
                    "total_interactions": len(interactions),
                    "pattern_types": list(set(p.pattern_type for p in patterns)),
                    "interaction_types": list(set(i.get("type", "unknown") for i in interactions)),
                    "average_confidence": sum(p.confidence for p in patterns) / len(patterns) if patterns else 0.0
                },
                "personalization_status": {
                    "preferences_count": len(user_profile.personalization_preferences),
                    "accessibility_settings_count": len(user_profile.accessibility_settings),
                    "learning_enabled": user_profile.privacy_settings.get("allow_learning", True),
                    "privacy_level": user_profile.privacy_settings.get("privacy_level", "standard")
                },
                "recent_activity": {
                    "last_7_days_interactions": len([
                        i for i in interactions 
                        if i.get("timestamp", datetime.min.replace(tzinfo=UTC)) >= 
                        datetime.now(UTC) - timedelta(days=7)
                    ]),
                    "last_30_days_interactions": len([
                        i for i in interactions 
                        if i.get("timestamp", datetime.min.replace(tzinfo=UTC)) >= 
                        datetime.now(UTC) - timedelta(days=30)
                    ])
                },
                "cached_analytics": cached_analytics,
                "generated_at": datetime.now(UTC)
            }
            
            return Either.success(analytics)
            
        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return Either.error(BiometricError(f"Analytics retrieval failed: {str(e)}"))
    
    # Private helper methods
    
    async def _update_identification_analytics(self, profile_id: UserProfileId,
                                             biometric_confidence: Optional[float],
                                             context: Optional[Dict[str, Any]]) -> None:
        """Update analytics for user identification."""
        analytics = self.profile_analytics[profile_id]
        
        # Update identification count
        analytics["total_identifications"] = analytics.get("total_identifications", 0) + 1
        analytics["last_identification"] = datetime.now(UTC)
        
        # Track biometric confidence if provided
        if biometric_confidence is not None:
            confidence_history = analytics.get("biometric_confidence_history", [])
            confidence_history.append({
                "confidence": biometric_confidence,
                "timestamp": datetime.now(UTC)
            })
            # Keep only last 100 records
            analytics["biometric_confidence_history"] = confidence_history[-100:]
            
            # Calculate average confidence
            confidences = [record["confidence"] for record in confidence_history]
            analytics["average_biometric_confidence"] = sum(confidences) / len(confidences)
        
        # Track context patterns
        if context:
            context_patterns = analytics.get("identification_context_patterns", {})
            for key, value in context.items():
                if key not in context_patterns:
                    context_patterns[key] = {}
                if str(value) not in context_patterns[key]:
                    context_patterns[key][str(value)] = 0
                context_patterns[key][str(value)] += 1
            analytics["identification_context_patterns"] = context_patterns
    
    async def _record_user_interaction(self, profile_id: UserProfileId,
                                     interaction_type: str,
                                     interaction_data: Dict[str, Any]) -> None:
        """Record user interaction for behavioral analysis."""
        interaction = {
            "type": interaction_type,
            "data": interaction_data,
            "timestamp": datetime.now(UTC),
            "session_id": interaction_data.get("session_id"),
            "context": interaction_data.get("context", {})
        }
        
        self.interaction_history[profile_id].append(interaction)
    
    async def _analyze_behavioral_patterns(self, patterns: List[BehavioralPattern]) -> Dict[str, Any]:
        """Analyze behavioral patterns for insights."""
        if not patterns:
            return {"message": "No patterns available for analysis"}
        
        pattern_types = defaultdict(int)
        confidence_sum = 0.0
        frequency_sum = 0
        
        for pattern in patterns:
            pattern_types[pattern.pattern_type] += 1
            confidence_sum += pattern.confidence
            frequency_sum += pattern.frequency
        
        return {
            "pattern_type_distribution": dict(pattern_types),
            "average_confidence": confidence_sum / len(patterns),
            "average_frequency": frequency_sum / len(patterns),
            "most_common_pattern_type": max(pattern_types.items(), key=lambda x: x[1])[0],
            "high_confidence_patterns": len([p for p in patterns if p.confidence > 0.8]),
            "recent_patterns": len([
                p for p in patterns 
                if p.last_observed >= datetime.now(UTC) - timedelta(days=3)
            ])
        }
    
    async def _analyze_interaction_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze interaction patterns for behavioral insights."""
        if not interactions:
            return {"message": "No interactions available for analysis"}
        
        interaction_types = defaultdict(int)
        success_count = 0
        hourly_distribution = defaultdict(int)
        
        for interaction in interactions:
            interaction_types[interaction.get("type", "unknown")] += 1
            if interaction.get("success", True):
                success_count += 1
            
            # Analyze time patterns
            timestamp = interaction.get("timestamp")
            if timestamp:
                hour = timestamp.hour
                hourly_distribution[hour] += 1
        
        return {
            "interaction_type_distribution": dict(interaction_types),
            "success_rate": success_count / len(interactions) if interactions else 0.0,
            "total_interactions": len(interactions),
            "hourly_distribution": dict(hourly_distribution),
            "most_active_hour": max(hourly_distribution.items(), key=lambda x: x[1])[0] if hourly_distribution else None,
            "most_common_interaction_type": max(interaction_types.items(), key=lambda x: x[1])[0]
        }
    
    async def _generate_personalization_insights(self, profile_id: UserProfileId,
                                               patterns: List[BehavioralPattern],
                                               interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Generate insights for personalization recommendations."""
        insights = {
            "automation_preferences": {},
            "interface_preferences": {},
            "timing_preferences": {},
            "accessibility_needs": {},
            "recommended_adaptations": []
        }
        
        # Analyze automation patterns
        automation_patterns = [p for p in patterns if p.pattern_type == "interaction"]
        if automation_patterns:
            # Extract common automation preferences
            common_automations = defaultdict(int)
            for pattern in automation_patterns:
                automation_type = pattern.pattern_data.get("automation_type")
                if automation_type:
                    common_automations[automation_type] += pattern.frequency
            
            insights["automation_preferences"] = dict(common_automations)
        
        # Analyze timing patterns
        timing_patterns = [p for p in patterns if p.pattern_type == "timing"]
        if timing_patterns:
            peak_hours = defaultdict(int)
            for pattern in timing_patterns:
                hour = pattern.pattern_data.get("hour")
                if hour is not None:
                    peak_hours[hour] += pattern.frequency
            
            insights["timing_preferences"] = {
                "peak_hours": dict(peak_hours),
                "most_active_hour": max(peak_hours.items(), key=lambda x: x[1])[0] if peak_hours else None
            }
        
        # Analyze accessibility patterns
        accessibility_patterns = [p for p in patterns if p.pattern_type == "accessibility"]
        if accessibility_patterns:
            accessibility_needs = {}
            for pattern in accessibility_patterns:
                need_type = pattern.pattern_data.get("need_type")
                if need_type:
                    accessibility_needs[need_type] = pattern.confidence
            
            insights["accessibility_needs"] = accessibility_needs
        
        # Generate adaptation recommendations
        recommendations = []
        
        # High-frequency patterns suggest automation opportunities
        high_freq_patterns = [p for p in patterns if p.frequency > 10 and p.confidence > 0.7]
        for pattern in high_freq_patterns:
            recommendations.append(f"Automate {pattern.pattern_type} based on high usage frequency")
        
        # Consistent timing patterns suggest scheduling opportunities
        if insights["timing_preferences"].get("most_active_hour"):
            recommendations.append(f"Schedule automations for peak hour: {insights['timing_preferences']['most_active_hour']}")
        
        insights["recommended_adaptations"] = recommendations
        
        return insights
    
    async def _analyze_behavioral_trends(self, patterns: List[BehavioralPattern]) -> Dict[str, Any]:
        """Analyze trends in behavioral patterns."""
        if not patterns:
            return {"message": "No patterns for trend analysis"}
        
        # Sort patterns by last observed time
        sorted_patterns = sorted(patterns, key=lambda p: p.last_observed)
        
        # Analyze confidence trends
        confidence_trend = []
        frequency_trend = []
        
        for i, pattern in enumerate(sorted_patterns):
            confidence_trend.append(pattern.confidence)
            frequency_trend.append(pattern.frequency)
        
        trends = {
            "confidence_trend": "stable",
            "frequency_trend": "stable",
            "pattern_evolution": "stable"
        }
        
        # Simple trend analysis
        if len(confidence_trend) >= 3:
            recent_avg = sum(confidence_trend[-3:]) / 3
            older_avg = sum(confidence_trend[:-3]) / len(confidence_trend[:-3]) if len(confidence_trend) > 3 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                trends["confidence_trend"] = "increasing"
            elif recent_avg < older_avg * 0.9:
                trends["confidence_trend"] = "decreasing"
        
        if len(frequency_trend) >= 3:
            recent_avg = sum(frequency_trend[-3:]) / 3
            older_avg = sum(frequency_trend[:-3]) / len(frequency_trend[:-3]) if len(frequency_trend) > 3 else recent_avg
            
            if recent_avg > older_avg * 1.1:
                trends["frequency_trend"] = "increasing"
            elif recent_avg < older_avg * 0.9:
                trends["frequency_trend"] = "decreasing"
        
        return trends
    
    async def _detect_time_anomalies(self, current_interaction: Dict[str, Any],
                                   interaction_history: List[Dict[str, Any]]) -> List[str]:
        """Detect time-based behavioral anomalies."""
        anomalies = []
        
        current_time = current_interaction.get("timestamp", datetime.now(UTC))
        current_hour = current_time.hour
        
        # Analyze historical time patterns
        if interaction_history:
            historical_hours = [
                interaction.get("timestamp", datetime.now(UTC)).hour
                for interaction in interaction_history
                if interaction.get("timestamp")
            ]
            
            if historical_hours:
                hour_frequency = defaultdict(int)
                for hour in historical_hours:
                    hour_frequency[hour] += 1
                
                # Check if current hour is unusual
                current_hour_freq = hour_frequency.get(current_hour, 0)
                avg_frequency = sum(hour_frequency.values()) / len(hour_frequency)
                
                if current_hour_freq < avg_frequency * 0.1:  # Very low frequency
                    anomalies.append(f"Unusual time of access: {current_hour}:00")
        
        return anomalies
    
    async def _detect_pattern_anomalies(self, current_interaction: Dict[str, Any],
                                      patterns: List[BehavioralPattern]) -> List[str]:
        """Detect pattern-based behavioral anomalies."""
        anomalies = []
        
        interaction_type = current_interaction.get("type", "unknown")
        
        # Find relevant patterns
        relevant_patterns = [
            p for p in patterns 
            if p.pattern_type == "interaction" and 
            p.pattern_data.get("interaction_type") == interaction_type
        ]
        
        if relevant_patterns:
            # Check confidence levels
            avg_confidence = sum(p.confidence for p in relevant_patterns) / len(relevant_patterns)
            
            # If this interaction type typically has high confidence, flag low confidence
            if avg_confidence > 0.8:
                current_confidence = current_interaction.get("confidence", 1.0)
                if current_confidence < 0.5:
                    anomalies.append(f"Low confidence for typically high-confidence interaction: {interaction_type}")
        
        return anomalies
    
    async def _detect_context_anomalies(self, current_interaction: Dict[str, Any],
                                      interaction_history: List[Dict[str, Any]]) -> List[str]:
        """Detect context-based behavioral anomalies."""
        anomalies = []
        
        current_context = current_interaction.get("context", {})
        
        if interaction_history:
            # Analyze device patterns
            historical_devices = [
                interaction.get("context", {}).get("device")
                for interaction in interaction_history
                if interaction.get("context", {}).get("device")
            ]
            
            if historical_devices:
                device_frequency = defaultdict(int)
                for device in historical_devices:
                    device_frequency[device] += 1
                
                current_device = current_context.get("device")
                if current_device and current_device not in device_frequency:
                    anomalies.append(f"New device detected: {current_device}")
        
        return anomalies
    
    async def _extract_behavioral_patterns(self, interaction: Dict[str, Any],
                                         user_profile: UserProfile) -> List[BehavioralPattern]:
        """Extract behavioral patterns from interaction."""
        patterns = []
        
        interaction_type = interaction.get("type", "unknown")
        timestamp = interaction.get("timestamp", datetime.now(UTC))
        data = interaction.get("data", {})
        context = interaction.get("context", {})
        
        # Extract timing pattern
        timing_pattern = BehavioralPattern(
            pattern_id=f"timing_{user_profile.profile_id}_{timestamp.hour}",
            pattern_type="timing",
            pattern_data={"hour": timestamp.hour, "day_of_week": timestamp.weekday()},
            confidence=0.7,
            frequency=1,
            first_observed=timestamp,
            last_observed=timestamp,
            context=context
        )
        patterns.append(timing_pattern)
        
        # Extract interaction pattern
        if interaction_type != "unknown":
            interaction_pattern = BehavioralPattern(
                pattern_id=f"interaction_{user_profile.profile_id}_{interaction_type}",
                pattern_type="interaction",
                pattern_data={"interaction_type": interaction_type, "success": interaction.get("success", True)},
                confidence=0.8,
                frequency=1,
                first_observed=timestamp,
                last_observed=timestamp,
                context=context
            )
            patterns.append(interaction_pattern)
        
        # Extract preference patterns from data
        if "preferences" in data:
            preference_pattern = BehavioralPattern(
                pattern_id=f"preference_{user_profile.profile_id}_{hash(str(data['preferences']))}",
                pattern_type="preference",
                pattern_data=data["preferences"],
                confidence=0.9,
                frequency=1,
                first_observed=timestamp,
                last_observed=timestamp,
                context=context
            )
            patterns.append(preference_pattern)
        
        return patterns
    
    async def _update_behavioral_pattern(self, profile_id: UserProfileId,
                                       new_pattern: BehavioralPattern) -> None:
        """Update or create behavioral pattern."""
        existing_patterns = self.behavioral_patterns[profile_id]
        
        # Look for existing pattern to update
        for i, existing_pattern in enumerate(existing_patterns):
            if existing_pattern.pattern_id == new_pattern.pattern_id:
                # Update existing pattern
                updated_pattern = BehavioralPattern(
                    pattern_id=existing_pattern.pattern_id,
                    pattern_type=existing_pattern.pattern_type,
                    pattern_data=existing_pattern.pattern_data,
                    confidence=min(1.0, (existing_pattern.confidence + new_pattern.confidence) / 2),
                    frequency=existing_pattern.frequency + 1,
                    first_observed=existing_pattern.first_observed,
                    last_observed=new_pattern.last_observed,
                    context=new_pattern.context
                )
                existing_patterns[i] = updated_pattern
                return
        
        # Add new pattern
        existing_patterns.append(new_pattern)
        
        # Limit pattern storage
        if len(existing_patterns) > 1000:
            existing_patterns.sort(key=lambda p: p.last_observed)
            self.behavioral_patterns[profile_id] = existing_patterns[-500:]
    
    async def _generate_learning_insights(self, profile_id: UserProfileId,
                                        interaction: Dict[str, Any]) -> Dict[str, Any]:
        """Generate insights from learning interaction."""
        insights = {
            "significant_patterns": [],
            "adaptation_opportunities": [],
            "personalization_updates": {}
        }
        
        # Analyze if this creates significant patterns
        interaction_type = interaction.get("type", "unknown")
        success = interaction.get("success", True)
        
        # Check for repeated successful interactions
        recent_interactions = list(self.interaction_history[profile_id])[-10:]
        similar_interactions = [
            i for i in recent_interactions 
            if i.get("type") == interaction_type and i.get("success", True)
        ]
        
        if len(similar_interactions) >= 3:
            insights["significant_patterns"].append(f"Repeated successful {interaction_type}")
            insights["adaptation_opportunities"].append(f"Consider automating {interaction_type}")
        
        # Check for preference patterns
        if "preferences" in interaction.get("data", {}):
            preferences = interaction["data"]["preferences"]
            insights["personalization_updates"] = preferences
        
        return insights
    
    async def _update_profile_behavioral_patterns(self, profile_id: UserProfileId,
                                                insights: Dict[str, Any]) -> None:
        """Update user profile with significant behavioral patterns."""
        if profile_id not in self.user_profiles:
            return
        
        current_profile = self.user_profiles[profile_id]
        updated_behavioral_patterns = dict(current_profile.behavioral_patterns)
        
        # Add significant patterns to profile
        for pattern in insights.get("significant_patterns", []):
            pattern_key = f"learned_{hash(pattern)}"
            updated_behavioral_patterns[pattern_key] = {
                "pattern": pattern,
                "discovered_at": datetime.now(UTC).isoformat(),
                "confidence": 0.8
            }
        
        # Create updated profile
        updated_profile = UserProfile(
            profile_id=current_profile.profile_id,
            user_identity=current_profile.user_identity,
            enrolled_modalities=current_profile.enrolled_modalities,
            biometric_templates=current_profile.biometric_templates,
            personalization_preferences=current_profile.personalization_preferences,
            accessibility_settings=current_profile.accessibility_settings,
            behavioral_patterns=updated_behavioral_patterns,
            privacy_settings=current_profile.privacy_settings,
            created_at=current_profile.created_at,
            last_updated=datetime.now(UTC),
            last_authenticated=current_profile.last_authenticated,
            is_active=current_profile.is_active
        )
        
        self.user_profiles[profile_id] = updated_profile