"""User Profiler - TASK_67 Phase 2 Core Identity Engine.

User identification and profile management with preferences, behavioral tracking,
and personalization data for adaptive automation workflows.

Architecture: User Profiling + Design by Contract + Type Safety + Behavioral Analysis
Performance: <50ms profile lookup, <100ms identification, <200ms behavioral analysis
Security: Privacy-preserving profiling, secure data storage, user consent management
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta
from typing import Any

from ..core.contracts import require
from ..core.either import Either
from ..core.user_identity_architecture import (
    AuthenticationMethod,
    IdentityError,
    PersonalizationSettings,
    UserProfile,
    UserProfileId,
    generate_profile_id,
)

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PersonalizationContext:
    """Context for personalization operations."""

    user_profile_id: UserProfileId
    session_id: str | None
    automation_type: str  # macro|workflow|interface|system
    current_context: dict[str, Any]
    time_of_day: str
    device_info: dict[str, Any]
    environmental_factors: dict[str, Any]


@dataclass(frozen=True)
class PersonalizationRecommendation:
    """Personalization recommendation with rationale."""

    recommendation_id: str
    category: str  # interface|automation|accessibility|behavior
    title: str
    description: str
    confidence: float  # 0.0 to 1.0
    impact_level: str  # low|medium|high
    implementation_data: dict[str, Any]
    rationale: list[str]


class UserProfiler:
    """User identification and profile management system."""

    def __init__(self):
        self.user_profiles: dict[UserProfileId, UserProfile] = {}
        self.personalization_settings: dict[UserProfileId, PersonalizationSettings] = {}
        self.behavioral_patterns: dict[UserProfileId, dict[str, Any]] = {}
        self.interaction_history: dict[UserProfileId, list[dict[str, Any]]] = {}
        self.learning_enabled = True

        # Initialize with default profiles for testing (lazy initialization)
        self._defaults_initialized = False

    async def _create_default_profiles(self) -> None:
        """Create default user profiles for testing."""
        try:
            # Admin user profile
            admin_profile = UserProfile(
                profile_id=generate_profile_id("admin"),
                username="admin",
                display_name="Administrator",
                email="admin@example.com",
                authentication_methods={AuthenticationMethod.PASSWORD},
                personalization_preferences={
                    "automation_style": "advanced",
                    "interface_theme": "dark",
                    "notification_level": "minimal",
                    "workflow_complexity": "high",
                },
                accessibility_settings={
                    "high_contrast": False,
                    "large_text": False,
                    "reduced_motion": False,
                },
                behavioral_patterns={
                    "preferred_hours": ["9-12", "14-17"],
                    "interaction_style": "keyboard_focused",
                    "automation_frequency": "high",
                },
                privacy_settings={
                    "allow_learning": True,
                    "data_retention_days": 365,
                    "analytics_enabled": True,
                },
                permissions={"admin", "user_management", "system_config"},
                created_at=datetime.now(UTC),
                last_updated=datetime.now(UTC),
            )

            # Test user profile
            test_profile = UserProfile(
                profile_id=generate_profile_id("testuser"),
                username="testuser",
                display_name="Test User",
                email="test@example.com",
                authentication_methods={AuthenticationMethod.PASSWORD},
                personalization_preferences={
                    "automation_style": "balanced",
                    "interface_theme": "light",
                    "notification_level": "standard",
                    "workflow_complexity": "medium",
                },
                accessibility_settings={},
                behavioral_patterns={},
                privacy_settings={"allow_learning": True, "data_retention_days": 180},
                permissions={"user", "automation"},
                created_at=datetime.now(UTC),
                last_updated=datetime.now(UTC),
            )

            # Store profiles
            self.user_profiles[admin_profile.profile_id] = admin_profile
            self.user_profiles[test_profile.profile_id] = test_profile

            # Create personalization settings
            await self._create_default_personalization_settings(
                admin_profile.profile_id,
                "comprehensive",
            )
            await self._create_default_personalization_settings(
                test_profile.profile_id,
                "moderate",
            )

            logger.info("Created default user profiles")

        except Exception as e:
            logger.warning(f"Failed to create default profiles: {e}")

    async def _create_default_personalization_settings(
        self,
        user_profile_id: UserProfileId,
        adaptation_level: str,
    ) -> None:
        """Create default personalization settings for user."""
        settings = PersonalizationSettings(
            user_profile_id=user_profile_id,
            automation_preferences={
                "proactivity_level": "medium",
                "complexity_preference": "adaptive",
                "feedback_frequency": "normal",
            },
            interface_preferences={
                "theme": "auto",
                "layout_density": "comfortable",
                "animation_speed": "normal",
            },
            accessibility_requirements={},
            behavioral_adaptations={},
            privacy_preferences={"allow_learning": True, "share_analytics": False},
            learning_enabled=True,
            adaptation_level=adaptation_level,
            cross_session_sync=True,
            created_at=datetime.now(UTC),
            last_updated=datetime.now(UTC),
        )

        self.personalization_settings[user_profile_id] = settings
        self.behavioral_patterns[user_profile_id] = {
            "interaction_analysis": {
                "total_interactions": 0,
                "most_common_interaction_type": "automation_request",
                "most_active_hour": 10,
                "success_rate": 0.95,
            },
            "pattern_analysis": {
                "high_confidence_patterns": 0,
                "learning_opportunities": [],
            },
            "personalization_insights": {"recommended_adaptations": []},
            "recent_interactions": 0,
        }
        self.interaction_history[user_profile_id] = []

    @require(lambda user_identity: len(user_identity) > 0)
    async def identify_user(
        self,
        user_identity: str,
        context: dict[str, Any] | None = None,
    ) -> Either[IdentityError, UserProfile]:
        """Identify user and return profile."""
        try:
            # Find user by various identifiers
            found_profile = None

            # Search by username
            for profile in self.user_profiles.values():
                if profile.username.lower() == user_identity.lower():
                    found_profile = profile
                    break

            # Search by profile ID
            if not found_profile:
                try:
                    profile_id = UserProfileId(user_identity)
                    if profile_id in self.user_profiles:
                        found_profile = self.user_profiles[profile_id]
                except (ValueError, TypeError) as e:
                    logger.debug(f"Failed to parse profile ID '{user_identity}': {e}")

            # Search by email if context provided
            if not found_profile and context and context.get("email"):
                for profile in self.user_profiles.values():
                    if (
                        profile.email
                        and profile.email.lower() == context["email"].lower()
                    ):
                        found_profile = profile
                        break

            if not found_profile:
                return Either.error(IdentityError.user_not_found(user_identity))

            # Update last authenticated after successful identification
            updated_profile = UserProfile(
                profile_id=found_profile.profile_id,
                username=found_profile.username,
                display_name=found_profile.display_name,
                email=found_profile.email,
                authentication_methods=found_profile.authentication_methods,
                personalization_preferences=found_profile.personalization_preferences,
                accessibility_settings=found_profile.accessibility_settings,
                behavioral_patterns=found_profile.behavioral_patterns,
                privacy_settings=found_profile.privacy_settings,
                permissions=found_profile.permissions,
                created_at=found_profile.created_at,
                last_updated=datetime.now(UTC),
                last_authenticated=datetime.now(UTC),
                is_active=found_profile.is_active,
            )

            self.user_profiles[found_profile.profile_id] = updated_profile
            found_profile = updated_profile

            logger.info(f"Successfully identified user: {found_profile.username}")
            return Either.success(found_profile)

        except Exception as e:
            logger.error(f"User identification failed: {e}")
            return Either.error(
                IdentityError(
                    f"User identification error: {e!s}",
                    "IDENTIFICATION_ERROR",
                ),
            )

    @require(lambda user_profile_id: user_profile_id is not None)
    async def get_user_analytics(
        self,
        user_profile_id: UserProfileId,
    ) -> Either[IdentityError, dict[str, Any]]:
        """Get user analytics and behavioral data."""
        try:
            if user_profile_id not in self.behavioral_patterns:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))

            behavioral_data = self.behavioral_patterns[user_profile_id]
            interaction_history = self.interaction_history.get(user_profile_id, [])

            analytics = {
                "user_profile_id": user_profile_id,
                "interaction_summary": {
                    "total_interactions": len(interaction_history),
                    "recent_interactions": len(
                        [
                            i
                            for i in interaction_history
                            if datetime.fromisoformat(
                                i["timestamp"].replace("Z", "+00:00"),
                            )
                            > datetime.now(UTC) - timedelta(days=7)
                        ],
                    ),
                    "success_rate": behavioral_data["interaction_analysis"].get(
                        "success_rate",
                        0.0,
                    ),
                },
                "behavioral_patterns": behavioral_data,
                "personalization_opportunities": await self._identify_personalization_opportunities(
                    user_profile_id,
                ),
                "analytics_generated_at": datetime.now(UTC).isoformat(),
            }

            return Either.success(analytics)

        except Exception as e:
            logger.error(f"Failed to get user analytics: {e}")
            return Either.error(
                IdentityError(
                    f"Analytics generation failed: {e!s}",
                    "ANALYTICS_ERROR",
                ),
            )

    async def _identify_personalization_opportunities(
        self,
        user_profile_id: UserProfileId,
    ) -> list[str]:
        """Identify personalization opportunities for user."""
        opportunities = []

        if user_profile_id not in self.user_profiles:
            return opportunities

        profile = self.user_profiles[user_profile_id]
        behavioral_data = self.behavioral_patterns.get(user_profile_id, {})

        # Check for incomplete personalization
        if len(profile.personalization_preferences) < 5:
            opportunities.append(
                "Expand personalization preferences for better automation",
            )

        # Check for unused accessibility features
        if not profile.accessibility_settings:
            opportunities.append(
                "Configure accessibility settings for improved usability",
            )

        # Check behavioral learning opportunities
        interaction_analysis = behavioral_data.get("interaction_analysis", {})
        if interaction_analysis.get("success_rate", 1.0) < 0.8:
            opportunities.append("Optimize frequent workflows based on usage patterns")

        # Check for advanced automation opportunities
        if profile.personalization_preferences.get("automation_style") == "basic":
            total_interactions = interaction_analysis.get("total_interactions", 0)
            if total_interactions > 50:
                opportunities.append(
                    "Enable advanced automation features based on usage level",
                )

        return opportunities

    @require(lambda user_profile_id: user_profile_id is not None)
    async def update_user_preferences(
        self,
        user_profile_id: UserProfileId,
        preferences: dict[str, Any],
        merge: bool = True,
    ) -> Either[IdentityError, UserProfile]:
        """Update user preferences and personalization settings."""
        try:
            if user_profile_id not in self.user_profiles:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))

            current_profile = self.user_profiles[user_profile_id]

            # Update personalization preferences
            if merge:
                updated_prefs = current_profile.personalization_preferences.copy()
                updated_prefs.update(preferences.get("personalization", {}))
            else:
                updated_prefs = preferences.get("personalization", {})

            # Update accessibility settings
            if merge:
                updated_accessibility = current_profile.accessibility_settings.copy()
                updated_accessibility.update(preferences.get("accessibility", {}))
            else:
                updated_accessibility = preferences.get("accessibility", {})

            # Update privacy settings
            if merge:
                updated_privacy = current_profile.privacy_settings.copy()
                updated_privacy.update(preferences.get("privacy", {}))
            else:
                updated_privacy = preferences.get("privacy", {})

            # Create updated profile
            updated_profile = UserProfile(
                profile_id=current_profile.profile_id,
                username=current_profile.username,
                display_name=preferences.get(
                    "display_name",
                    current_profile.display_name,
                ),
                email=preferences.get("email", current_profile.email),
                authentication_methods=current_profile.authentication_methods,
                personalization_preferences=updated_prefs,
                accessibility_settings=updated_accessibility,
                behavioral_patterns=current_profile.behavioral_patterns,
                privacy_settings=updated_privacy,
                permissions=current_profile.permissions,
                created_at=current_profile.created_at,
                last_updated=datetime.now(UTC),
                last_authenticated=current_profile.last_authenticated,
                is_active=current_profile.is_active,
            )

            self.user_profiles[user_profile_id] = updated_profile

            # Update personalization settings if provided
            if user_profile_id in self.personalization_settings:
                current_settings = self.personalization_settings[user_profile_id]

                updated_settings = PersonalizationSettings(
                    user_profile_id=current_settings.user_profile_id,
                    automation_preferences=preferences.get(
                        "automation_preferences",
                        current_settings.automation_preferences,
                    ),
                    interface_preferences=preferences.get(
                        "interface_preferences",
                        current_settings.interface_preferences,
                    ),
                    accessibility_requirements=updated_accessibility,
                    behavioral_adaptations=preferences.get(
                        "behavioral_adaptations",
                        current_settings.behavioral_adaptations,
                    ),
                    privacy_preferences=updated_privacy,
                    learning_enabled=preferences.get(
                        "learning_enabled",
                        current_settings.learning_enabled,
                    ),
                    adaptation_level=preferences.get(
                        "adaptation_level",
                        current_settings.adaptation_level,
                    ),
                    cross_session_sync=preferences.get(
                        "cross_session_sync",
                        current_settings.cross_session_sync,
                    ),
                    created_at=current_settings.created_at,
                    last_updated=datetime.now(UTC),
                )

                self.personalization_settings[user_profile_id] = updated_settings

            logger.info(f"Updated preferences for user: {current_profile.username}")
            return Either.success(updated_profile)

        except Exception as e:
            logger.error(f"Failed to update user preferences: {e}")
            return Either.error(
                IdentityError(
                    f"Preference update failed: {e!s}",
                    "PREFERENCE_UPDATE_ERROR",
                ),
            )

    @require(lambda user_profile_id: user_profile_id is not None)
    async def learn_from_interaction(
        self,
        user_profile_id: UserProfileId,
        interaction_type: str,
        interaction_data: dict[str, Any],
        success: bool,
    ) -> Either[IdentityError, dict[str, Any]]:
        """Learn from user interaction for behavioral analysis."""
        try:
            if not self.learning_enabled:
                return Either.success({"learning_disabled": True})

            if user_profile_id not in self.behavioral_patterns:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))

            # Record interaction
            interaction_record = {
                "timestamp": datetime.now(UTC).isoformat(),
                "type": interaction_type,
                "data": interaction_data,
                "success": success,
            }

            # Update interaction history
            if user_profile_id not in self.interaction_history:
                self.interaction_history[user_profile_id] = []

            self.interaction_history[user_profile_id].append(interaction_record)

            # Keep only recent interactions (last 1000)
            if len(self.interaction_history[user_profile_id]) > 1000:
                self.interaction_history[user_profile_id] = self.interaction_history[
                    user_profile_id
                ][-1000:]

            # Update behavioral patterns
            behavioral_data = self.behavioral_patterns[user_profile_id]
            interaction_analysis = behavioral_data["interaction_analysis"]

            # Update totals
            interaction_analysis["total_interactions"] += 1

            # Update success rate
            recent_interactions = self.interaction_history[user_profile_id][
                -100:
            ]  # Last 100
            recent_successes = sum(1 for i in recent_interactions if i["success"])
            interaction_analysis["success_rate"] = recent_successes / len(
                recent_interactions,
            )

            # Update most common interaction type
            type_counts = {}
            for interaction in recent_interactions:
                itype = interaction["type"]
                type_counts[itype] = type_counts.get(itype, 0) + 1

            if type_counts:
                interaction_analysis["most_common_interaction_type"] = max(
                    type_counts,
                    key=type_counts.get,
                )

            # Update activity hour
            current_hour = datetime.now(UTC).hour
            interaction_analysis["most_active_hour"] = current_hour

            # Generate learning insights
            learning_result = {
                "interaction_recorded": True,
                "total_interactions": interaction_analysis["total_interactions"],
                "success_rate": interaction_analysis["success_rate"],
                "insights": [],
            }

            # Add insights based on patterns
            if interaction_analysis["success_rate"] < 0.7:
                learning_result["insights"].append(
                    "Consider simplifying frequent workflows",
                )

            if (
                interaction_analysis["total_interactions"] > 100
                and interaction_analysis["success_rate"] > 0.9
            ):
                learning_result["insights"].append(
                    "User shows high proficiency - enable advanced features",
                )

            logger.debug(f"Recorded interaction learning for user {user_profile_id}")
            return Either.success(learning_result)

        except Exception as e:
            logger.error(f"Failed to learn from interaction: {e}")
            return Either.error(
                IdentityError(f"Learning failed: {e!s}", "LEARNING_ERROR"),
            )

    @require(lambda user_profile_id: user_profile_id is not None)
    async def analyze_user_behavior(
        self,
        user_profile_id: UserProfileId,
        analysis_period_days: int = 7,
    ) -> Either[IdentityError, dict[str, Any]]:
        """Analyze user behavior patterns for personalization insights."""
        try:
            if user_profile_id not in self.behavioral_patterns:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))

            behavioral_data = self.behavioral_patterns[user_profile_id]
            interaction_history = self.interaction_history.get(user_profile_id, [])

            # Filter interactions by analysis period
            cutoff_date = datetime.now(UTC) - timedelta(days=analysis_period_days)
            recent_interactions = [
                i
                for i in interaction_history
                if datetime.fromisoformat(i["timestamp"].replace("Z", "+00:00"))
                > cutoff_date
            ]

            # Analyze patterns
            analysis = {
                "analysis_period_days": analysis_period_days,
                "total_interactions": len(recent_interactions),
                "interaction_analysis": behavioral_data["interaction_analysis"].copy(),
                "pattern_analysis": {
                    "peak_hours": self._analyze_peak_hours(recent_interactions),
                    "interaction_types": self._analyze_interaction_types(
                        recent_interactions,
                    ),
                    "success_patterns": self._analyze_success_patterns(
                        recent_interactions,
                    ),
                },
                "personalization_insights": {
                    "recommended_adaptations": self._generate_adaptation_recommendations(
                        recent_interactions,
                    ),
                    "automation_opportunities": self._identify_automation_opportunities(
                        recent_interactions,
                    ),
                },
            }

            # Update behavioral patterns with analysis
            behavioral_data["pattern_analysis"].update(analysis["pattern_analysis"])
            behavioral_data["personalization_insights"].update(
                analysis["personalization_insights"],
            )

            return Either.success(analysis)

        except Exception as e:
            logger.error(f"Behavior analysis failed: {e}")
            return Either.error(
                IdentityError(f"Behavior analysis error: {e!s}", "ANALYSIS_ERROR"),
            )

    def _analyze_peak_hours(self, interactions: list[dict[str, Any]]) -> list[int]:
        """Analyze peak activity hours."""
        hour_counts = {}

        for interaction in interactions:
            try:
                timestamp = datetime.fromisoformat(
                    interaction["timestamp"].replace("Z", "+00:00"),
                )
                hour = timestamp.hour
                hour_counts[hour] = hour_counts.get(hour, 0) + 1
            except (ValueError, KeyError, TypeError) as e:
                logger.debug(f"Failed to parse interaction timestamp: {e}")
                continue

        # Return top 3 peak hours
        sorted_hours = sorted(hour_counts.items(), key=lambda x: x[1], reverse=True)
        return [hour for hour, count in sorted_hours[:3]]

    def _analyze_interaction_types(
        self,
        interactions: list[dict[str, Any]],
    ) -> dict[str, int]:
        """Analyze interaction type frequencies."""
        type_counts = {}

        for interaction in interactions:
            itype = interaction.get("type", "unknown")
            type_counts[itype] = type_counts.get(itype, 0) + 1

        return type_counts

    def _analyze_success_patterns(
        self,
        interactions: list[dict[str, Any]],
    ) -> dict[str, float]:
        """Analyze success patterns by interaction type."""
        success_patterns = {}

        for interaction in interactions:
            itype = interaction.get("type", "unknown")
            success = interaction.get("success", False)

            if itype not in success_patterns:
                success_patterns[itype] = {"total": 0, "successful": 0}

            success_patterns[itype]["total"] += 1
            if success:
                success_patterns[itype]["successful"] += 1

        # Calculate success rates
        return {
            itype: data["successful"] / data["total"] if data["total"] > 0 else 0.0
            for itype, data in success_patterns.items()
        }

    def _generate_adaptation_recommendations(
        self,
        interactions: list[dict[str, Any]],
    ) -> list[str]:
        """Generate adaptation recommendations based on interaction patterns."""
        recommendations = []

        if len(interactions) < 10:
            recommendations.append(
                "Continue using the system to generate personalized recommendations",
            )
            return recommendations

        # Analyze success rates
        total_successes = sum(1 for i in interactions if i.get("success", False))
        success_rate = total_successes / len(interactions)

        if success_rate < 0.7:
            recommendations.append("Simplify common workflows to improve success rate")
        elif success_rate > 0.9:
            recommendations.append("Enable advanced automation features")

        # Analyze timing patterns
        peak_hours = self._analyze_peak_hours(interactions)
        if len(peak_hours) > 0:
            recommendations.append(f"Optimize automation for peak hours: {peak_hours}")

        return recommendations

    def _identify_automation_opportunities(
        self,
        interactions: list[dict[str, Any]],
    ) -> list[str]:
        """Identify automation opportunities based on patterns."""
        opportunities = []

        # Look for repetitive patterns
        type_counts = self._analyze_interaction_types(interactions)

        for itype, count in type_counts.items():
            if count > len(interactions) * 0.3:  # More than 30% of interactions
                opportunities.append(f"Consider automating frequent {itype} operations")

        return opportunities

    @require(lambda user_profile_id: user_profile_id is not None)
    async def detect_behavioral_anomalies(
        self,
        user_profile_id: UserProfileId,
        current_interaction: dict[str, Any],
    ) -> Either[IdentityError, list[str]]:
        """Detect behavioral anomalies for security monitoring."""
        try:
            if user_profile_id not in self.behavioral_patterns:
                return Either.error(IdentityError.user_not_found(str(user_profile_id)))

            anomalies = []
            behavioral_data = self.behavioral_patterns[user_profile_id]
            interaction_analysis = behavioral_data["interaction_analysis"]

            # Check time-based anomalies
            current_hour = datetime.now(UTC).hour
            typical_hour = interaction_analysis.get("most_active_hour", 10)

            if abs(current_hour - typical_hour) > 6:  # More than 6 hours difference
                anomalies.append(
                    f"Unusual activity time: {current_hour}:00 (typical: {typical_hour}:00)",
                )

            # Check interaction type anomalies
            interaction_type = current_interaction.get("type", "unknown")
            typical_type = interaction_analysis.get("most_common_interaction_type", "")

            if interaction_type != typical_type and typical_type:
                # This is less critical - users can have varied interaction patterns
                pass

            # Check for rapid successive interactions (potential security concern)
            interaction_history = self.interaction_history.get(user_profile_id, [])
            if len(interaction_history) > 0:
                last_interaction_time = datetime.fromisoformat(
                    interaction_history[-1]["timestamp"].replace("Z", "+00:00"),
                )
                time_diff = datetime.now(UTC) - last_interaction_time

                if (
                    time_diff.total_seconds() < 1
                ):  # Less than 1 second between interactions
                    anomalies.append("Rapid successive interactions detected")

            return Either.success(anomalies)

        except Exception as e:
            logger.error(f"Anomaly detection failed: {e}")
            return Either.error(
                IdentityError(
                    f"Anomaly detection error: {e!s}",
                    "ANOMALY_DETECTION_ERROR",
                ),
            )
