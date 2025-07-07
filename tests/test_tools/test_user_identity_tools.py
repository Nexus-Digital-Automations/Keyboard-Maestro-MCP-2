"""Test Suite for User Identity Tools - TASK_67 Validation.

Comprehensive tests for username-based user identity management, authentication,
personalization, and adaptive automation tools.

Test Coverage:
- User authentication with username/password
- User identification and profile retrieval
- Personalization and adaptive automation
- User profile management with privacy compliance
- Behavioral analysis and pattern recognition
- Multi-user context switching
- Security validation and error handling

Testing Approach: Property-based + Integration + Security validation
"""

from __future__ import annotations

import asyncio
import secrets  # S311 fix: Import secure random for cryptographic purposes
import string
from datetime import UTC, datetime, timedelta
from typing import Any
from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from src.core.either import Either

# Import the underlying identity system
from src.core.user_identity_architecture import (
    AuthenticationMethod,
    SecurityLevel,
)

# Import the tools we're testing - now they are plain async functions like core tools
from src.server.tools.user_identity_tools import (
    km_analyze_user_behavior,
    km_authenticate_user,
    km_identify_user,
    km_manage_user_profiles,
    km_personalize_automation,
    km_switch_user_context,
)


@pytest.fixture
def mock_context() -> Any:
    """Create a mock FastMCP Context for testing."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    return context


@pytest.fixture
def sample_user_data() -> Any:
    """Sample user data for testing."""
    return {
        "username": "testuser",
        "password": "TestPassword123!",
        "display_name": "Test User",
        "email": "test@example.com",
        "profile_data": {
            "personalization": {
                "automation_style": "balanced",
                "interface_theme": "dark",
                "notification_level": "standard",
            },
            "accessibility": {"high_contrast": False, "large_text": False},
            "privacy": {"allow_learning": True, "data_retention_days": 180},
        },
    }


class TestKMAuthenticateUser:
    """Test suite for km_authenticate_user tool."""

    @pytest.mark.asyncio
    async def test_authenticate_user_success_password(
        self,
        mock_context: Any,
        sample_user_data: Any,
    ) -> None:
        """Test successful password authentication."""
        # Create mock AuthenticationResult object
        mock_auth_result = Mock()
        mock_auth_result.session_id = "test-session-123"
        mock_auth_result.success = True
        mock_auth_result.user_profile_id = "test-profile-456"
        mock_auth_result.username = sample_user_data["username"]
        mock_auth_result.authentication_method = AuthenticationMethod.PASSWORD
        mock_auth_result.security_level = SecurityLevel.MEDIUM
        mock_auth_result.session_token = "test-token-789"  # noqa: S105 - Test authentication token
        mock_auth_result.processing_time_ms = 50.0
        mock_auth_result.authenticated_at = datetime.now(UTC)
        mock_auth_result.expires_at = datetime.now(UTC) + timedelta(hours=8)
        mock_auth_result.permissions = {"read", "write"}
        mock_auth_result.security_warnings = []
        mock_auth_result.create_session = lambda: {"session_data": "mock"}

        # Mock session manager to return success
        mock_session_result = Mock()
        mock_session_result.is_success.return_value = True
        mock_session_result.value = Mock()
        mock_session_result.value.isolation_level = "moderate"

        with (
            patch(
                "src.server.tools.user_identity_tools.authentication_manager",
            ) as mock_auth_manager,
            patch(
                "src.server.tools.user_identity_tools.session_manager",
            ) as mock_session_manager,
        ):
            mock_auth_manager.authenticate_user = AsyncMock(
                return_value=Either.success(mock_auth_result),
            )
            mock_session_manager.create_session = AsyncMock(
                return_value=mock_session_result,
            )

            result = await km_authenticate_user(
                username=sample_user_data["username"],
                authentication_method="password",
                password=sample_user_data["password"],
                security_level="medium",
                session_duration=8,
                remember_session=True,
                ctx=mock_context,
            )

        assert result["success"]
        assert "session_id" in result
        assert "user_profile_id" in result
        assert result["username"] == sample_user_data["username"]
        assert result["authentication_method"] == "password"
        assert result["security_level"] == "medium"
        assert "session_token" in result
        assert "expires_at" in result
        assert "permissions" in result
        assert result["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_method(self, mock_context: Any) -> None:
        """Test authentication with invalid method."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="invalid_method",
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "INVALID_AUTH_METHOD"
        assert "supported_methods" in result
        assert "password" in result["supported_methods"]

    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_security_level(self, mock_context: Any) -> None:
        """Test authentication with invalid security level."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            password="test123",  # noqa: S106 - Test authentication data
            security_level="invalid_level",
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "INVALID_SECURITY_LEVEL"
        assert "supported_levels" in result

    @pytest.mark.asyncio
    async def test_authenticate_user_missing_password(self, mock_context: Any) -> None:
        """Test password authentication without password."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "PASSWORD_REQUIRED"

    @pytest.mark.asyncio
    async def test_authenticate_user_session_method(self, mock_context: Any) -> None:
        """Test session-based authentication."""
        result = await km_authenticate_user(
            username="admin",  # Use admin user that should exist
            authentication_method="session",
            security_level="low",
            ctx=mock_context,
        )

        # Session auth may fail if no existing session, but should handle gracefully
        assert "success" in result
        assert "error_code" in result or result["success"]

    @pytest.mark.asyncio
    async def test_authenticate_user_timeout_validation(self, mock_context: Any) -> None:
        """Test timeout parameter validation."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            password="test123",  # noqa: S106 - Test authentication data
            timeout=500,  # Beyond max limit
            ctx=mock_context,
        )

        # Should still work, but timeout should be clamped
        assert "success" in result
        assert "processing_time_ms" in result


class TestKMIdentifyUser:
    """Test suite for km_identify_user tool."""

    @pytest.mark.asyncio
    async def test_identify_user_success(self, mock_context: Any) -> None:
        """Test successful user identification."""
        identification_context = {"username": "admin", "source": "test_context"}

        # Create mock user profile
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-123"
        mock_user_profile.username = "admin"
        mock_user_profile.display_name = "Admin User"
        mock_user_profile.personalization_preferences = {
            "automation_style": "advanced",
            "interface_theme": "dark",
        }
        mock_user_profile.accessibility_settings = {
            "high_contrast": False,
            "large_text": False,
        }
        mock_user_profile.privacy_settings = {
            "allow_learning": True,
            "data_retention_days": 180,
        }
        mock_user_profile.permissions = {"admin", "macro_creation", "system_control"}
        mock_user_profile.created_at = datetime.now(UTC)
        mock_user_profile.last_updated = datetime.now(UTC)
        mock_user_profile.last_authenticated = datetime.now(UTC)
        mock_user_profile.is_active = True

        # Create mock analytics result
        mock_analytics_result = Mock()
        mock_analytics_result.is_success.return_value = True
        mock_analytics_result.value = {"pattern_data": "test"}

        with patch(
            "src.server.tools.user_identity_tools.user_profiler",
        ) as mock_user_profiler:
            # Mock successful identification
            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_user_profiler.get_user_analytics = AsyncMock(
                return_value=mock_analytics_result,
            )

            result = await km_identify_user(
                identification_context=identification_context,
                include_preferences=True,
                load_behavioral_data=True,
                privacy_level="standard",
                ctx=mock_context,
            )

        assert result["success"]
        assert result["username"] == "admin"
        assert "user_profile_id" in result
        assert "display_name" in result
        assert result["identification_confidence"] == 1.0
        assert "preferences" in result
        assert "permissions" in result
        assert "profile_metadata" in result

    @pytest.mark.asyncio
    async def test_identify_user_missing_identity(self, mock_context: Any) -> None:
        """Test identification with missing user identity."""
        identification_context = {
            "source": "test_context",
            # Missing username/user_id
        }

        result = await km_identify_user(
            identification_context=identification_context,
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "MISSING_IDENTITY"
        assert "required_fields" in result

    @pytest.mark.asyncio
    async def test_identify_user_invalid_privacy_level(self, mock_context: Any) -> None:
        """Test identification with invalid privacy level."""
        identification_context = {"username": "admin"}

        result = await km_identify_user(
            identification_context=identification_context,
            privacy_level="invalid_level",
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "INVALID_PRIVACY_LEVEL"
        assert "supported_levels" in result

    @pytest.mark.asyncio
    async def test_identify_user_privacy_levels(self, mock_context: Any) -> None:
        """Test different privacy levels."""
        identification_context = {"username": "admin"}

        # Test minimal privacy level
        result_minimal = await km_identify_user(
            identification_context=identification_context,
            include_preferences=True,
            privacy_level="minimal",
            ctx=mock_context,
        )

        if result_minimal["success"]:
            # Minimal privacy should exclude preferences
            assert (
                "preferences" not in result_minimal or not result_minimal["preferences"]
            )

        # Test enhanced privacy level
        result_enhanced = await km_identify_user(
            identification_context=identification_context,
            load_behavioral_data=True,
            privacy_level="enhanced",
            ctx=mock_context,
        )

        if result_enhanced["success"]:
            # Enhanced privacy should include behavioral data
            assert (
                "behavioral_patterns" in result_enhanced
                or result_enhanced.get("behavioral_patterns") is not None
            )

    @pytest.mark.asyncio
    async def test_identify_user_not_found(self, mock_context: Any) -> None:
        """Test identification of non-existent user."""
        identification_context = {"username": "nonexistent_user_12345"}

        result = await km_identify_user(
            identification_context=identification_context,
            ctx=mock_context,
        )

        assert not result["success"]
        assert "error_code" in result


class TestKMPersonalizeAutomation:
    """Test suite for km_personalize_automation tool."""

    @pytest.mark.asyncio
    async def test_personalize_automation_success(self, mock_context: Any) -> None:
        """Test successful automation personalization."""
        # Create mock user profile
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-123"
        mock_user_profile.username = "admin"

        # Create mock adaptation result
        mock_adaptation_result = Mock()
        mock_adaptation_result.success = True
        mock_adaptation_result.adaptations_applied = [
            "interface_theme",
            "automation_style",
        ]
        mock_adaptation_result.user_experience_score = 8.5
        mock_adaptation_result.performance_impact = {
            "response_time": "+5ms",
            "accuracy": "+12%",
        }
        mock_adaptation_result.user_feedback_required = False
        mock_adaptation_result.settings = {
            "learning_mode": True,
            "adaptation_level": "moderate",
            "personalization_scope": ["preferences", "behavior"],
        }

        with (
            patch(
                "src.server.tools.user_identity_tools.user_profiler",
            ) as mock_user_profiler,
            patch(
                "src.server.tools.user_identity_tools.personalization_engine",
            ) as mock_personalization_engine,
        ):
            # Mock successful identification and personalization
            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_personalization_engine.personalize_automation = AsyncMock(
                return_value=Either.success(mock_adaptation_result),
            )

            result = await km_personalize_automation(
                user_identity="admin",
                automation_context="macro",
                personalization_scope=["preferences", "behavior"],
                adaptation_level="moderate",
                learning_mode=True,
                ctx=mock_context,
            )

        assert result["success"]
        assert result["automation_context"] == "macro"
        assert result["adaptation_level"] == "moderate"
        assert "adaptations_applied" in result
        assert "user_experience_score" in result
        assert "performance_impact" in result
        assert "settings" in result
        assert result["settings"]["learning_mode"]

    @pytest.mark.asyncio
    async def test_personalize_automation_different_levels(self, mock_context: Any) -> None:
        """Test different adaptation levels."""
        adaptation_levels = ["light", "moderate", "comprehensive"]

        for level in adaptation_levels:
            result = await km_personalize_automation(
                user_identity="admin",
                automation_context="workflow",
                adaptation_level=level,
                ctx=mock_context,
            )

            assert "success" in result
            if result["success"]:
                assert result["adaptation_level"] == level

    @pytest.mark.asyncio
    async def test_personalize_automation_user_not_found(self, mock_context: Any) -> None:
        """Test personalization for non-existent user."""
        result = await km_personalize_automation(
            user_identity="nonexistent_user_98765",
            automation_context="interface",
            ctx=mock_context,
        )

        assert not result["success"]
        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_personalize_automation_contexts(self, mock_context: Any) -> None:
        """Test different automation contexts."""
        contexts = ["macro", "workflow", "interface", "system"]

        for context in contexts:
            result = await km_personalize_automation(
                user_identity="admin",
                automation_context=context,
                ctx=mock_context,
            )

            assert "success" in result
            if result["success"]:
                assert result["automation_context"] == context


class TestKMManageUserProfiles:
    """Test suite for km_manage_user_profiles tool."""

    @pytest.mark.asyncio
    async def test_manage_profiles_list_operation(self, mock_context: Any) -> None:
        """Test listing user profiles."""
        result = await km_manage_user_profiles(
            operation="list",
            user_identity="admin",
            ctx=mock_context,
        )

        assert result["success"]
        assert result["operation"] == "list"
        assert "profiles" in result
        assert len(result["profiles"]) > 0

        # Check profile structure
        for profile in result["profiles"]:
            assert "username" in profile
            assert "display_name" in profile
            assert "profile_id" in profile

    @pytest.mark.asyncio
    async def test_manage_profiles_update_operation(self, mock_context: Any) -> None:
        """Test updating user preferences."""
        preferences = {
            "personalization": {
                "automation_style": "advanced",
                "interface_theme": "light",
            },
            "accessibility": {"high_contrast": True},
        }

        # Create mock user profile
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-123"
        mock_user_profile.username = "admin"

        # Create mock update result
        mock_update_result = Mock()
        mock_update_result.success = True
        mock_update_result.updated_profile = mock_user_profile
        mock_update_result.preferences_updated = ["personalization", "accessibility"]
        mock_update_result.validation_results = {"status": "valid", "warnings": []}

        with patch(
            "src.server.tools.user_identity_tools.user_profiler",
        ) as mock_user_profiler:
            # Mock successful operations
            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_user_profiler.update_user_preferences = AsyncMock(
                return_value=Either.success(mock_update_result),
            )

            result = await km_manage_user_profiles(
                operation="update",
                user_identity="admin",
                preferences=preferences,
                ctx=mock_context,
            )

        assert result["operation"] == "update"
        if result["success"]:
            assert "updated_profile" in result
            assert "preferences_updated" in result

    @pytest.mark.asyncio
    async def test_manage_profiles_delete_operation(self, mock_context: Any) -> None:
        """Test deleting user data."""
        # Create mock user profile
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-123"
        mock_user_profile.username = "testuser"

        # Create mock delete result
        mock_delete_result = Mock()
        mock_delete_result.success = True
        mock_delete_result.compliance_status = "GDPR_COMPLIANT"
        mock_delete_result.audit_logged = True
        mock_delete_result.deletion_summary = {
            "profiles_deleted": 1,
            "data_purged": True,
        }

        with (
            patch(
                "src.server.tools.user_identity_tools.user_profiler",
            ) as mock_user_profiler,
            patch(
                "src.server.tools.user_identity_tools.privacy_manager",
            ) as mock_privacy_manager,
        ):
            # Mock successful operations
            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_privacy_manager.delete_user_data = AsyncMock(
                return_value=Either.success(mock_delete_result),
            )

            result = await km_manage_user_profiles(
                operation="delete",
                user_identity="testuser",
                compliance_mode=True,
                audit_logging=True,
                ctx=mock_context,
            )

        assert result["operation"] == "delete"
        # Delete may succeed or fail depending on user existence
        assert "success" in result
        if result["success"]:
            assert result["compliance_status"] == "GDPR_COMPLIANT"
            assert result["audit_logged"]

    @pytest.mark.asyncio
    async def test_manage_profiles_invalid_operation(self, mock_context: Any) -> None:
        """Test invalid operation."""
        result = await km_manage_user_profiles(
            operation="invalid_operation",
            user_identity="admin",
            ctx=mock_context,
        )

        assert not result["success"]
        assert result["error_code"] == "INVALID_OPERATION"
        assert "valid_operations" in result

    @pytest.mark.asyncio
    async def test_manage_profiles_security_compliance(self, mock_context: Any) -> None:
        """Test security and compliance features."""
        result = await km_manage_user_profiles(
            operation="list",
            user_identity="admin",
            encryption_level="high",
            compliance_mode=True,
            audit_logging=True,
            ctx=mock_context,
        )

        assert "security" in result
        assert result["security"]["encryption_level"] == "high"
        assert result["security"]["data_encrypted"]

        assert "compliance" in result
        assert result["compliance"]["gdpr_compliant"]
        assert result["compliance"]["audit_enabled"]


class TestKMAnalyzeUserBehavior:
    """Test suite for km_analyze_user_behavior tool."""

    @pytest.mark.asyncio
    async def test_analyze_behavior_success(self, mock_context: Any) -> None:
        """Test successful behavior analysis."""
        # Create mock user profile
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-123"
        mock_user_profile.username = "admin"

        # Create mock behavior analysis result
        mock_analysis_result = Mock()
        mock_analysis_result.success = True
        mock_analysis_result.analysis_period = "week"
        mock_analysis_result.period_days = 7
        mock_analysis_result.behavior_analysis = {
            "usage_patterns": {"most_active_day": "Tuesday", "avg_daily_actions": 45},
            "preferences": {"automation_style": "advanced", "interface_theme": "dark"},
            "timing": {"peak_usage": "09:00-11:00", "lowest_usage": "15:00-17:00"},
        }
        mock_analysis_result.predictions = {"next_week_usage": 47}
        mock_analysis_result.anomalies_detected = []
        mock_analysis_result.privacy_compliant = True

        with (
            patch(
                "src.server.tools.user_identity_tools.user_profiler",
            ) as mock_user_profiler,
            patch(
                "src.server.tools.user_identity_tools.personalization_engine",
            ) as mock_personalization_engine,
        ):
            # Mock successful operations
            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_user_profiler.analyze_user_behavior = AsyncMock(
                return_value=Either.success(mock_analysis_result),
            )
            mock_user_profiler.detect_behavioral_anomalies = AsyncMock(
                return_value=Either.success([]),
            )
            mock_personalization_engine.get_personalization_insights = AsyncMock(
                return_value=Either.success({"recent_adaptations": []}),
            )

            result = await km_analyze_user_behavior(
                user_identity="admin",
                analysis_period="week",
                behavior_patterns=["usage", "preferences", "timing"],
                include_predictions=True,
                anomaly_detection=True,
                privacy_preserving=True,
                ctx=mock_context,
            )

        assert result["success"]
        assert result["analysis_period"] == "week"
        assert result["period_days"] == 7
        assert "behavior_analysis" in result
        assert "patterns_analyzed" in result
        assert "insights" in result
        assert "predictions" in result
        assert "privacy" in result
        assert result["privacy"]["privacy_preserving"]

    @pytest.mark.asyncio
    async def test_analyze_behavior_different_periods(self, mock_context: Any) -> None:
        """Test different analysis periods."""
        periods = ["day", "week", "month", "custom"]
        expected_days = [1, 7, 30, 14]  # custom defaults to 14

        for period, expected in zip(periods, expected_days, strict=False):
            result = await km_analyze_user_behavior(
                user_identity="admin",
                analysis_period=period,
                ctx=mock_context,
            )

            if result["success"]:
                assert result["period_days"] == expected

    @pytest.mark.asyncio
    async def test_analyze_behavior_user_not_found(self, mock_context: Any) -> None:
        """Test behavior analysis for non-existent user."""
        result = await km_analyze_user_behavior(
            user_identity="nonexistent_user_54321",
            ctx=mock_context,
        )

        assert not result["success"]
        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_analyze_behavior_features(self, mock_context: Any) -> None:
        """Test different analysis features."""
        result = await km_analyze_user_behavior(
            user_identity="admin",
            include_predictions=True,
            anomaly_detection=True,
            generate_insights=True,
            adaptive_recommendations=True,
            ctx=mock_context,
        )

        if result["success"]:
            assert "predictions" in result
            assert result["predictions"]["enabled"]
            assert "anomalies_detected" in result
            assert "recommendations" in result
            assert result["recommendations"]["adaptive_enabled"]


class TestKMSwitchUserContext:
    """Test suite for km_switch_user_context tool."""

    @pytest.mark.asyncio
    async def test_switch_context_success(self, mock_context: Any) -> None:
        """Test successful user context switching."""
        # Create mock user profiles
        mock_current_profile = Mock()
        mock_current_profile.profile_id = "current-profile-123"
        mock_current_profile.username = "admin"

        mock_target_profile = Mock()
        mock_target_profile.profile_id = "target-profile-456"
        mock_target_profile.username = "testuser"
        mock_target_profile.display_name = "Test User"
        mock_target_profile.personalization_preferences = {
            "automation_style": "balanced",
        }
        mock_target_profile.accessibility_settings = {"high_contrast": False}
        mock_target_profile.privacy_settings = {"allow_learning": True}
        mock_target_profile.permissions = ["read", "execute"]

        # Create mock switch result
        mock_switch_result = Mock()
        mock_switch_result.session_id = "new-session-789"
        mock_switch_result.isolation_level = "moderate"
        mock_switch_result.user_profile_id = "target-profile-456"
        mock_switch_result.environment_data = {"context": "desktop"}
        mock_switch_result.switch_summary = {"switched_at": "2025-07-04T22:00:00Z"}

        with (
            patch(
                "src.server.tools.user_identity_tools.user_profiler",
            ) as mock_user_profiler,
            patch(
                "src.server.tools.user_identity_tools.session_manager",
            ) as mock_session_manager,
        ):
            # Mock successful operations
            mock_user_profiler.identify_user = AsyncMock(
                side_effect=[
                    Either.success(mock_target_profile),  # First call for target user
                    Either.success(
                        mock_current_profile,
                    ),  # Second call for current user (if provided)
                ],
            )
            mock_session_manager.switch_user_context = AsyncMock(
                return_value=Either.success(mock_switch_result),
            )

            result = await km_switch_user_context(
                target_user="testuser",
                current_user="admin",
                preserve_session=True,
                load_preferences=True,
                security_validation=True,
                audit_switch=True,
                ctx=mock_context,
            )

        # Context switch may fail if users don't have sessions, but should handle gracefully
        assert "success" in result
        assert result["target_user"] == "testuser"
        assert result["current_user"] == "admin"

        if result["success"]:
            assert "context_switch" in result
            assert "new_context" in result
            assert "security" in result
            assert result["security"]["validation_performed"]
            assert result["security"]["audit_logged"]

    @pytest.mark.asyncio
    async def test_switch_context_target_not_found(self, mock_context: Any) -> None:
        """Test context switch with non-existent target user."""
        result = await km_switch_user_context(
            target_user="nonexistent_user_99999",
            ctx=mock_context,
        )

        assert not result["success"]
        assert "error_code" in result

    @pytest.mark.asyncio
    async def test_switch_context_features(self, mock_context: Any) -> None:
        """Test different context switch features."""
        result = await km_switch_user_context(
            target_user="admin",
            preserve_session=False,
            load_preferences=False,
            security_validation=False,
            audit_switch=False,
            ctx=mock_context,
        )

        assert "success" in result
        if result["success"]:
            context_switch = result.get("context_switch", {})
            security = result.get("security", {})

            assert not context_switch.get("context_preserved")
            assert not context_switch.get("preferences_loaded")
            assert not security.get("validation_performed")
            assert not security.get("audit_logged")


class TestUserIdentityToolsSecurity:
    """Security-focused tests for user identity tools."""

    @pytest.mark.asyncio
    async def test_authentication_sql_injection_prevention(self, mock_context: Any) -> None:
        """Test SQL injection prevention in authentication."""
        malicious_usernames = [
            "admin'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "admin'/**/OR/**/1=1",
            "'; DELETE FROM users; --",
        ]

        for malicious_username in malicious_usernames:
            # S106 fix: Use variable instead of hardcoded password in test
            test_password = "test123"  # noqa: S105 - Test credential, not production
            result = await km_authenticate_user(
                username=malicious_username,
                authentication_method="password",
                password=test_password,  # noqa: S106 - Test authentication data
                ctx=mock_context,
            )

            # Should handle malicious input gracefully without exposing system errors
            assert "success" in result
            if not result["success"]:
                assert "error_code" in result
                # Should not contain database error messages
                error_msg = result.get("error", "").lower()
                assert "database" not in error_msg
                assert "sql" not in error_msg
                assert "table" not in error_msg

    @pytest.mark.asyncio
    async def test_identification_xss_prevention(self, mock_context: Any) -> None:
        """Test XSS prevention in user identification."""
        malicious_contexts = [
            {"username": "<script>alert('xss')</script>"},
            {"username": "admin<img src=x onerror=alert(1)>"},
            {"username": "javascript:alert('xss')"},
            {"email": "<script>document.location='http://evil.com'</script>"},
        ]

        for malicious_context in malicious_contexts:
            result = await km_identify_user(
                identification_context=malicious_context,
                ctx=mock_context,
            )

            # Should handle malicious input without executing scripts
            assert "success" in result
            if "username" in result:
                # Should not contain unescaped script tags
                assert "<script>" not in result["username"]
                assert "javascript:" not in result["username"]

    @pytest.mark.asyncio
    async def test_profile_management_path_traversal_prevention(self, mock_context: Any) -> None:
        """Test path traversal prevention in profile management."""
        malicious_user_identities = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM",
        ]

        for malicious_identity in malicious_user_identities:
            result = await km_manage_user_profiles(
                operation="list",
                user_identity=malicious_identity,
                ctx=mock_context,
            )

            # Should handle malicious paths without accessing system files
            assert "success" in result
            if not result["success"]:
                error_msg = result.get("error", "").lower()
                # Should not contain system file paths or access errors
                assert "etc/passwd" not in error_msg
                assert "system32" not in error_msg
                assert "permission denied" not in error_msg

    @pytest.mark.asyncio
    async def test_privacy_level_enforcement(self, mock_context: Any) -> None:
        """Test privacy level enforcement across tools."""
        # Test that minimal privacy level restricts data access
        minimal_result = await km_identify_user(
            identification_context={"username": "admin"},
            privacy_level="minimal",
            include_preferences=True,
            load_behavioral_data=True,
            ctx=mock_context,
        )

        if minimal_result["success"]:
            # Minimal privacy should exclude sensitive data
            preferences = minimal_result.get("preferences", {})
            behavioral = minimal_result.get("behavioral_patterns")

            # Should have limited or no preference data in minimal mode
            assert not preferences or len(preferences) <= 2
            assert behavioral is None or not behavioral


class TestUserIdentityToolsPerformance:
    """Performance-focused tests for user identity tools."""

    @pytest.mark.asyncio
    async def test_authentication_performance(self, mock_context: Any) -> None:
        """Test authentication performance requirements."""
        start_time = datetime.now(UTC)

        # S106 fix: Use variable instead of hardcoded password in test
        test_admin_password = "SecureAdmin123!"  # noqa: S105 - Test credential, not production
        result = await km_authenticate_user(
            username="admin",
            authentication_method="password",
            password=test_admin_password,  # noqa: S106 - Test authentication data
            ctx=mock_context,
        )

        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds() * 1000

        # Should complete within reasonable time (< 1000ms for test environment)
        assert execution_time < 1000

        if result["success"]:
            # Should report processing time
            assert "processing_time_ms" in result
            assert result["processing_time_ms"] > 0

    @pytest.mark.asyncio
    async def test_identification_performance(self, mock_context: Any) -> None:
        """Test identification performance requirements."""
        start_time = datetime.now(UTC)

        result = await km_identify_user(
            identification_context={"username": "admin"},
            load_behavioral_data=True,
            ctx=mock_context,
        )

        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds() * 1000

        # Should complete within reasonable time
        assert execution_time < 500

        if result["success"]:
            assert "processing_time_ms" in result

    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_context: Any) -> None:
        """Test concurrent user identity operations."""
        # Create multiple concurrent authentication requests
        tasks = []
        for _i in range(5):
            task = km_authenticate_user(
                username="admin",
                authentication_method="session",
                ctx=mock_context,
            )
            tasks.append(task)

        # Execute all tasks concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)

        # All operations should complete without exceptions
        for result in results:
            assert not isinstance(result, Exception)
            assert "success" in result


class TestUserIdentityToolsIntegration:
    """Integration tests for user identity tool workflows."""

    @pytest.mark.asyncio
    async def test_complete_identity_workflow(self, mock_context: Any, sample_user_data: Any) -> None:
        """Test complete user identity workflow."""
        # Create comprehensive mocks for all identity components
        mock_user_profile = Mock()
        mock_user_profile.profile_id = "test-profile-workflow"
        mock_user_profile.username = sample_user_data["username"]
        mock_user_profile.display_name = sample_user_data["display_name"]
        mock_user_profile.personalization_preferences = sample_user_data[
            "profile_data"
        ]["personalization"]
        mock_user_profile.accessibility_settings = sample_user_data["profile_data"][
            "accessibility"
        ]
        mock_user_profile.privacy_settings = sample_user_data["profile_data"]["privacy"]
        mock_user_profile.permissions = ["read", "execute"]
        mock_user_profile.created_at = datetime.now(UTC)
        mock_user_profile.last_updated = datetime.now(UTC)
        mock_user_profile.last_authenticated = datetime.now(UTC)
        mock_user_profile.is_active = True

        mock_auth_result = Mock()
        mock_auth_result.session_id = "workflow-session-123"
        mock_auth_result.user_profile_id = "test-profile-workflow"
        mock_auth_result.username = sample_user_data["username"]
        mock_auth_result.authentication_method = AuthenticationMethod.PASSWORD
        mock_auth_result.security_level = SecurityLevel.MEDIUM
        # S105 fix: Use variable instead of hardcoded token in test
        test_session_token = "token123"  # noqa: S105 - Test token, not production
        mock_auth_result.session_token = test_session_token  # noqa: S105 - Test session data
        mock_auth_result.expires_at = datetime.now(UTC) + timedelta(hours=8)
        mock_auth_result.permissions = ["read", "execute"]
        mock_auth_result.processing_time_ms = 45.0
        mock_auth_result.security_warnings = []

        mock_session_context = Mock()
        mock_session_context.isolation_level = "moderate"

        mock_adaptation_result = Mock()
        mock_adaptation_result.success = True
        mock_adaptation_result.adaptations_applied = [
            "interface_theme",
            "automation_style",
        ]
        mock_adaptation_result.user_experience_score = 0.85
        mock_adaptation_result.performance_impact = "minimal"
        mock_adaptation_result.user_feedback_required = False
        mock_adaptation_result.next_learning_opportunities = []

        mock_analysis_data = {
            "usage_patterns": {"most_active_day": "Tuesday", "avg_daily_actions": 45},
            "personalization_insights": {
                "automation_opportunities": [],
                "recommended_adaptations": [],
            },
        }

        mock_updated_profile = Mock()
        mock_updated_profile.profile_id = "test-profile-workflow"
        mock_updated_profile.username = sample_user_data["username"]
        mock_updated_profile.display_name = sample_user_data["display_name"]
        mock_updated_profile.last_updated = datetime.now(UTC)

        with (
            patch(
                "src.server.tools.user_identity_tools.authentication_manager",
            ) as mock_auth_manager,
            patch(
                "src.server.tools.user_identity_tools.user_profiler",
            ) as mock_user_profiler,
            patch(
                "src.server.tools.user_identity_tools.session_manager",
            ) as mock_session_manager,
            patch(
                "src.server.tools.user_identity_tools.personalization_engine",
            ) as mock_personalization_engine,
        ):
            # Set up all mocks for the workflow
            mock_auth_manager.authenticate_user = AsyncMock(
                return_value=Either.success(mock_auth_result),
            )
            mock_session_manager.create_session = AsyncMock(
                return_value=Either.success(mock_session_context),
            )

            mock_user_profiler.identify_user = AsyncMock(
                return_value=Either.success(mock_user_profile),
            )
            mock_user_profiler.get_user_analytics = AsyncMock(
                return_value=Either.success({}),
            )
            mock_user_profiler.analyze_user_behavior = AsyncMock(
                return_value=Either.success(mock_analysis_data),
            )
            mock_user_profiler.detect_behavioral_anomalies = AsyncMock(
                return_value=Either.success([]),
            )
            mock_user_profiler.update_user_preferences = AsyncMock(
                return_value=Either.success(mock_updated_profile),
            )

            mock_personalization_engine.personalize_automation = AsyncMock(
                return_value=Either.success(mock_adaptation_result),
            )
            mock_personalization_engine.get_personalization_insights = AsyncMock(
                return_value=Either.success({"recent_adaptations": []}),
            )

            # Step 1: Authenticate user
            auth_result = await km_authenticate_user(
                username=sample_user_data["username"],
                authentication_method="password",
                password=sample_user_data["password"],
                ctx=mock_context,
            )

            assert auth_result["success"]
            username = auth_result["username"]

            # Step 2: Identify user and get profile
            identify_result = await km_identify_user(
                identification_context={"username": username},
                include_preferences=True,
                ctx=mock_context,
            )

            assert identify_result["success"]
            identify_result["user_profile_id"]

            # Step 3: Personalize automation
            personalize_result = await km_personalize_automation(
                user_identity=username,
                automation_context="workflow",
                adaptation_level="moderate",
                ctx=mock_context,
            )

            assert personalize_result["success"]

            # Step 4: Analyze behavior
            behavior_result = await km_analyze_user_behavior(
                user_identity=username,
                analysis_period="week",
                ctx=mock_context,
            )

            assert behavior_result["success"]

            # Step 5: Update preferences
            update_result = await km_manage_user_profiles(
                operation="update",
                user_identity=username,
                preferences={"personalization": {"automation_style": "advanced"}},
                ctx=mock_context,
            )

            # Update may succeed or fail, but should handle gracefully
            assert "success" in update_result

    @pytest.mark.asyncio
    async def test_multi_user_context_workflow(self, mock_context: Any) -> None:
        """Test multi-user context management workflow."""
        # Switch from admin to test user
        switch_result = await km_switch_user_context(
            target_user="testuser",
            current_user="admin",
            preserve_session=True,
            load_preferences=True,
            ctx=mock_context,
        )

        # Context switch may fail if users don't have sessions
        assert "success" in switch_result

        if switch_result["success"]:
            # Verify new context is loaded
            assert "new_context" in switch_result
            assert "user_profile_id" in switch_result["new_context"]

            # Switch back to admin
            switch_back = await km_switch_user_context(
                target_user="admin",
                current_user="testuser",
                preserve_session=True,
                ctx=mock_context,
            )

            assert "success" in switch_back

    @pytest.mark.asyncio
    async def test_privacy_workflow(self, mock_context: Any) -> None:
        """Test privacy-focused workflow."""
        # Identify user with enhanced privacy
        identify_result = await km_identify_user(
            identification_context={"username": "admin"},
            privacy_level="enhanced",
            load_behavioral_data=True,
            ctx=mock_context,
        )

        if identify_result["success"]:
            # Analyze behavior with privacy preservation
            behavior_result = await km_analyze_user_behavior(
                user_identity="admin",
                privacy_preserving=True,
                generate_insights=True,
                ctx=mock_context,
            )

            assert behavior_result["success"]
            assert behavior_result["privacy"]["privacy_preserving"]

            # Manage profile with compliance
            manage_result = await km_manage_user_profiles(
                operation="list",
                user_identity="admin",
                compliance_mode=True,
                audit_logging=True,
                ctx=mock_context,
            )

            assert manage_result["compliance"]["gdpr_compliant"]


# Property-based tests for robust validation
@pytest.mark.property
class TestUserIdentityToolsProperties:
    """Property-based tests for user identity tools."""

    @pytest.mark.asyncio
    async def test_authentication_input_validation_properties(self, mock_context: Any) -> None:
        """Property: Authentication should validate all inputs safely."""
        # S311 fix: Use secrets module for cryptographically secure random generation
        # (import moved to top for better organization)

        # Generate cryptographically secure random but safe test inputs
        test_cases = []
        for _ in range(10):
            username = "".join(
                secrets.choice(string.ascii_letters + string.digits)
                for _ in range(secrets.randbelow(50) + 1)
            )
            password = "".join(
                secrets.choice(string.ascii_letters + string.digits + "!@#$%")
                for _ in range(secrets.randbelow(93) + 8)
            )
            test_cases.append((username, password))

        for username, password in test_cases:
            result = await km_authenticate_user(
                username=username,
                authentication_method="password",
                password=password,
                ctx=mock_context,
            )

            # Property: All results should have success field
            assert "success" in result
            assert isinstance(result["success"], bool)

            # Property: Processing time should always be reported
            assert "processing_time_ms" in result
            assert isinstance(result["processing_time_ms"], int | float)
            assert result["processing_time_ms"] >= 0

    @pytest.mark.asyncio
    async def test_identification_context_properties(self, mock_context: Any) -> None:
        """Property: Identification should handle various context formats."""
        test_contexts = [
            {"username": "admin"},
            {"user_id": "admin"},
            {"email": "admin@example.com"},
            {"username": "admin", "source": "test"},
            {"username": "admin", "timestamp": datetime.now(UTC).isoformat()},
        ]

        for context in test_contexts:
            result = await km_identify_user(
                identification_context=context,
                ctx=mock_context,
            )

            # Property: All results should have consistent structure
            assert "success" in result
            assert "processing_time_ms" in result

            if result["success"]:
                # Property: Successful identification should have required fields
                assert "username" in result
                assert "user_profile_id" in result
                assert "identification_confidence" in result
                assert 0.0 <= result["identification_confidence"] <= 1.0
