"""
Test Suite for User Identity Tools - TASK_67 Validation

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

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
from unittest.mock import AsyncMock, Mock, patch
from typing import Dict, List, Any, Optional

import fastmcp
from fastmcp import Context

# Import the tools we're testing - now they are plain async functions like core tools
from src.server.tools.user_identity_tools import (
    km_authenticate_user,
    km_identify_user,
    km_personalize_automation,
    km_manage_user_profiles,
    km_analyze_user_behavior,
    km_switch_user_context
)

# Import the underlying identity system
from src.core.user_identity_architecture import (
    AuthenticationMethod, SecurityLevel, PrivacyLevel, IdentityError,
    AuthenticationRequest, UserProfile, PersonalizationSettings,
    UserProfileId, generate_profile_id, generate_session_id, generate_session_token,
    AuthenticationResult, UserSessionId, SessionToken
)
from src.core.either import Either
from src.identity.authentication_manager import IdentityAuthenticationManager
from src.identity.user_profiler import UserProfiler, PersonalizationContext
from src.identity.personalization_engine import PersonalizationEngine
from src.identity.privacy_manager import PrivacyManager, ConsentType
from src.identity.session_manager import SessionManager


@pytest.fixture
def mock_context():
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
def sample_user_data():
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
                "notification_level": "standard"
            },
            "accessibility": {
                "high_contrast": False,
                "large_text": False
            },
            "privacy": {
                "allow_learning": True,
                "data_retention_days": 180
            }
        }
    }


class TestKMAuthenticateUser:
    """Test suite for km_authenticate_user tool."""
    
    @pytest.mark.asyncio
    async def test_authenticate_user_success_password(self, mock_context, sample_user_data):
        """Test successful password authentication."""
        # Create mock AuthenticationResult object
        mock_auth_result = Mock()
        mock_auth_result.session_id = "test-session-123"
        mock_auth_result.success = True
        mock_auth_result.user_profile_id = "test-profile-456"
        mock_auth_result.username = sample_user_data["username"]
        mock_auth_result.authentication_method = AuthenticationMethod.PASSWORD
        mock_auth_result.security_level = SecurityLevel.MEDIUM
        mock_auth_result.session_token = "test-token-789"
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
        
        with patch('src.server.tools.user_identity_tools.authentication_manager') as mock_auth_manager, \
             patch('src.server.tools.user_identity_tools.session_manager') as mock_session_manager:
            
            mock_auth_manager.authenticate_user = AsyncMock(return_value=Either.success(mock_auth_result))
            mock_session_manager.create_session = AsyncMock(return_value=mock_session_result)
            
            result = await km_authenticate_user(
                username=sample_user_data["username"],
                authentication_method="password", 
                password=sample_user_data["password"],
                security_level="medium",
                session_duration=8,
                remember_session=True,
                ctx=mock_context
            )
        
        assert result["success"] == True
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
    async def test_authenticate_user_invalid_method(self, mock_context):
        """Test authentication with invalid method."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="invalid_method",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "INVALID_AUTH_METHOD"
        assert "supported_methods" in result
        assert "password" in result["supported_methods"]
    
    @pytest.mark.asyncio
    async def test_authenticate_user_invalid_security_level(self, mock_context):
        """Test authentication with invalid security level."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            password="test123",
            security_level="invalid_level",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "INVALID_SECURITY_LEVEL"
        assert "supported_levels" in result
    
    @pytest.mark.asyncio
    async def test_authenticate_user_missing_password(self, mock_context):
        """Test password authentication without password."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "PASSWORD_REQUIRED"
    
    @pytest.mark.asyncio
    async def test_authenticate_user_session_method(self, mock_context):
        """Test session-based authentication."""
        result = await km_authenticate_user(
            username="admin",  # Use admin user that should exist
            authentication_method="session",
            security_level="low",
            ctx=mock_context
        )
        
        # Session auth may fail if no existing session, but should handle gracefully
        assert "success" in result
        assert "error_code" in result or result["success"] == True
    
    @pytest.mark.asyncio
    async def test_authenticate_user_timeout_validation(self, mock_context):
        """Test timeout parameter validation."""
        result = await km_authenticate_user(
            username="testuser",
            authentication_method="password",
            password="test123",
            timeout=500,  # Beyond max limit
            ctx=mock_context
        )
        
        # Should still work, but timeout should be clamped
        assert "success" in result
        assert "processing_time_ms" in result


class TestKMIdentifyUser:
    """Test suite for km_identify_user tool."""
    
    @pytest.mark.asyncio
    async def test_identify_user_success(self, mock_context):
        """Test successful user identification."""
        identification_context = {
            "username": "admin",
            "source": "test_context"
        }
        
        result = await km_identify_user(
            identification_context=identification_context,
            include_preferences=True,
            load_behavioral_data=True,
            privacy_level="standard",
            ctx=mock_context
        )
        
        assert result["success"] == True
        assert result["username"] == "admin"
        assert "user_profile_id" in result
        assert "display_name" in result
        assert result["identification_confidence"] == 1.0
        assert "preferences" in result
        assert "permissions" in result
        assert "profile_metadata" in result
    
    @pytest.mark.asyncio
    async def test_identify_user_missing_identity(self, mock_context):
        """Test identification with missing user identity."""
        identification_context = {
            "source": "test_context"
            # Missing username/user_id
        }
        
        result = await km_identify_user(
            identification_context=identification_context,
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "MISSING_IDENTITY"
        assert "required_fields" in result
    
    @pytest.mark.asyncio
    async def test_identify_user_invalid_privacy_level(self, mock_context):
        """Test identification with invalid privacy level."""
        identification_context = {"username": "admin"}
        
        result = await km_identify_user(
            identification_context=identification_context,
            privacy_level="invalid_level",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "INVALID_PRIVACY_LEVEL"
        assert "supported_levels" in result
    
    @pytest.mark.asyncio
    async def test_identify_user_privacy_levels(self, mock_context):
        """Test different privacy levels."""
        identification_context = {"username": "admin"}
        
        # Test minimal privacy level
        result_minimal = await km_identify_user(
            identification_context=identification_context,
            include_preferences=True,
            privacy_level="minimal",
            ctx=mock_context
        )
        
        if result_minimal["success"]:
            # Minimal privacy should exclude preferences
            assert "preferences" not in result_minimal or not result_minimal["preferences"]
        
        # Test enhanced privacy level
        result_enhanced = await km_identify_user(
            identification_context=identification_context,
            load_behavioral_data=True,
            privacy_level="enhanced",
            ctx=mock_context
        )
        
        if result_enhanced["success"]:
            # Enhanced privacy should include behavioral data
            assert "behavioral_patterns" in result_enhanced or result_enhanced.get("behavioral_patterns") is not None
    
    @pytest.mark.asyncio
    async def test_identify_user_not_found(self, mock_context):
        """Test identification of non-existent user."""
        identification_context = {"username": "nonexistent_user_12345"}
        
        result = await km_identify_user(
            identification_context=identification_context,
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert "error_code" in result


class TestKMPersonalizeAutomation:
    """Test suite for km_personalize_automation tool."""
    
    @pytest.mark.asyncio
    async def test_personalize_automation_success(self, mock_context):
        """Test successful automation personalization."""
        result = await km_personalize_automation(
            user_identity="admin",
            automation_context="macro",
            personalization_scope=["preferences", "behavior"],
            adaptation_level="moderate",
            learning_mode=True,
            ctx=mock_context
        )
        
        assert result["success"] == True
        assert result["automation_context"] == "macro"
        assert result["adaptation_level"] == "moderate"
        assert "adaptations_applied" in result
        assert "user_experience_score" in result
        assert "performance_impact" in result
        assert "settings" in result
        assert result["settings"]["learning_mode"] == True
    
    @pytest.mark.asyncio
    async def test_personalize_automation_different_levels(self, mock_context):
        """Test different adaptation levels."""
        adaptation_levels = ["light", "moderate", "comprehensive"]
        
        for level in adaptation_levels:
            result = await km_personalize_automation(
                user_identity="admin",
                automation_context="workflow",
                adaptation_level=level,
                ctx=mock_context
            )
            
            assert "success" in result
            if result["success"]:
                assert result["adaptation_level"] == level
    
    @pytest.mark.asyncio
    async def test_personalize_automation_user_not_found(self, mock_context):
        """Test personalization for non-existent user."""
        result = await km_personalize_automation(
            user_identity="nonexistent_user_98765",
            automation_context="interface",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert "error_code" in result
    
    @pytest.mark.asyncio
    async def test_personalize_automation_contexts(self, mock_context):
        """Test different automation contexts."""
        contexts = ["macro", "workflow", "interface", "system"]
        
        for context in contexts:
            result = await km_personalize_automation(
                user_identity="admin",
                automation_context=context,
                ctx=mock_context
            )
            
            assert "success" in result
            if result["success"]:
                assert result["automation_context"] == context


class TestKMManageUserProfiles:
    """Test suite for km_manage_user_profiles tool."""
    
    @pytest.mark.asyncio
    async def test_manage_profiles_list_operation(self, mock_context):
        """Test listing user profiles."""
        result = await km_manage_user_profiles(
            operation="list",
            user_identity="admin",
            ctx=mock_context
        )
        
        assert result["success"] == True
        assert result["operation"] == "list"
        assert "profiles" in result
        assert len(result["profiles"]) > 0
        
        # Check profile structure
        for profile in result["profiles"]:
            assert "username" in profile
            assert "display_name" in profile
            assert "profile_id" in profile
    
    @pytest.mark.asyncio
    async def test_manage_profiles_update_operation(self, mock_context):
        """Test updating user preferences."""
        preferences = {
            "personalization": {
                "automation_style": "advanced",
                "interface_theme": "light"
            },
            "accessibility": {
                "high_contrast": True
            }
        }
        
        result = await km_manage_user_profiles(
            operation="update",
            user_identity="admin",
            preferences=preferences,
            ctx=mock_context
        )
        
        assert result["operation"] == "update"
        if result["success"]:
            assert "updated_profile" in result
            assert "preferences_updated" in result
    
    @pytest.mark.asyncio
    async def test_manage_profiles_delete_operation(self, mock_context):
        """Test deleting user data."""
        result = await km_manage_user_profiles(
            operation="delete",
            user_identity="testuser",
            compliance_mode=True,
            audit_logging=True,
            ctx=mock_context
        )
        
        assert result["operation"] == "delete"
        # Delete may succeed or fail depending on user existence
        assert "success" in result
        if result["success"]:
            assert result["compliance_status"] == "GDPR_COMPLIANT"
            assert result["audit_logged"] == True
    
    @pytest.mark.asyncio
    async def test_manage_profiles_invalid_operation(self, mock_context):
        """Test invalid operation."""
        result = await km_manage_user_profiles(
            operation="invalid_operation",
            user_identity="admin",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert result["error_code"] == "INVALID_OPERATION"
        assert "valid_operations" in result
    
    @pytest.mark.asyncio
    async def test_manage_profiles_security_compliance(self, mock_context):
        """Test security and compliance features."""
        result = await km_manage_user_profiles(
            operation="list",
            user_identity="admin",
            encryption_level="high",
            compliance_mode=True,
            audit_logging=True,
            ctx=mock_context
        )
        
        assert "security" in result
        assert result["security"]["encryption_level"] == "high"
        assert result["security"]["data_encrypted"] == True
        
        assert "compliance" in result
        assert result["compliance"]["gdpr_compliant"] == True
        assert result["compliance"]["audit_enabled"] == True


class TestKMAnalyzeUserBehavior:
    """Test suite for km_analyze_user_behavior tool."""
    
    @pytest.mark.asyncio
    async def test_analyze_behavior_success(self, mock_context):
        """Test successful behavior analysis."""
        result = await km_analyze_user_behavior(
            user_identity="admin",
            analysis_period="week",
            behavior_patterns=["usage", "preferences", "timing"],
            include_predictions=True,
            anomaly_detection=True,
            privacy_preserving=True,
            ctx=mock_context
        )
        
        assert result["success"] == True
        assert result["analysis_period"] == "week"
        assert result["period_days"] == 7
        assert "behavior_analysis" in result
        assert "patterns_analyzed" in result
        assert "insights" in result
        assert "predictions" in result
        assert "privacy" in result
        assert result["privacy"]["privacy_preserving"] == True
    
    @pytest.mark.asyncio
    async def test_analyze_behavior_different_periods(self, mock_context):
        """Test different analysis periods."""
        periods = ["day", "week", "month", "custom"]
        expected_days = [1, 7, 30, 14]  # custom defaults to 14
        
        for period, expected in zip(periods, expected_days):
            result = await km_analyze_user_behavior(
                user_identity="admin",
                analysis_period=period,
                ctx=mock_context
            )
            
            if result["success"]:
                assert result["period_days"] == expected
    
    @pytest.mark.asyncio
    async def test_analyze_behavior_user_not_found(self, mock_context):
        """Test behavior analysis for non-existent user."""
        result = await km_analyze_user_behavior(
            user_identity="nonexistent_user_54321",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert "error_code" in result
    
    @pytest.mark.asyncio
    async def test_analyze_behavior_features(self, mock_context):
        """Test different analysis features."""
        result = await km_analyze_user_behavior(
            user_identity="admin",
            include_predictions=True,
            anomaly_detection=True,
            generate_insights=True,
            adaptive_recommendations=True,
            ctx=mock_context
        )
        
        if result["success"]:
            assert "predictions" in result
            assert result["predictions"]["enabled"] == True
            assert "anomalies_detected" in result
            assert "recommendations" in result
            assert result["recommendations"]["adaptive_enabled"] == True


class TestKMSwitchUserContext:
    """Test suite for km_switch_user_context tool."""
    
    @pytest.mark.asyncio
    async def test_switch_context_success(self, mock_context):
        """Test successful user context switching."""
        result = await km_switch_user_context(
            target_user="testuser",
            current_user="admin",
            preserve_session=True,
            load_preferences=True,
            security_validation=True,
            audit_switch=True,
            ctx=mock_context
        )
        
        # Context switch may fail if users don't have sessions, but should handle gracefully
        assert "success" in result
        assert result["target_user"] == "testuser"
        assert result["current_user"] == "admin"
        
        if result["success"]:
            assert "context_switch" in result
            assert "new_context" in result
            assert "security" in result
            assert result["security"]["validation_performed"] == True
            assert result["security"]["audit_logged"] == True
    
    @pytest.mark.asyncio
    async def test_switch_context_target_not_found(self, mock_context):
        """Test context switch with non-existent target user."""
        result = await km_switch_user_context(
            target_user="nonexistent_user_99999",
            ctx=mock_context
        )
        
        assert result["success"] == False
        assert "error_code" in result
    
    @pytest.mark.asyncio
    async def test_switch_context_features(self, mock_context):
        """Test different context switch features."""
        result = await km_switch_user_context(
            target_user="admin",
            preserve_session=False,
            load_preferences=False,
            security_validation=False,
            audit_switch=False,
            ctx=mock_context
        )
        
        assert "success" in result
        if result["success"]:
            context_switch = result.get("context_switch", {})
            security = result.get("security", {})
            
            assert context_switch.get("context_preserved") == False
            assert context_switch.get("preferences_loaded") == False
            assert security.get("validation_performed") == False
            assert security.get("audit_logged") == False


class TestUserIdentityToolsSecurity:
    """Security-focused tests for user identity tools."""
    
    @pytest.mark.asyncio
    async def test_authentication_sql_injection_prevention(self, mock_context):
        """Test SQL injection prevention in authentication."""
        malicious_usernames = [
            "admin'; DROP TABLE users; --",
            "admin' OR '1'='1",
            "admin'/**/OR/**/1=1",
            "'; DELETE FROM users; --"
        ]
        
        for malicious_username in malicious_usernames:
            result = await km_authenticate_user(
                username=malicious_username,
                authentication_method="password",
                password="test123",
                ctx=mock_context
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
    async def test_identification_xss_prevention(self, mock_context):
        """Test XSS prevention in user identification."""
        malicious_contexts = [
            {"username": "<script>alert('xss')</script>"},
            {"username": "admin<img src=x onerror=alert(1)>"},
            {"username": "javascript:alert('xss')"},
            {"email": "<script>document.location='http://evil.com'</script>"}
        ]
        
        for malicious_context in malicious_contexts:
            result = await km_identify_user(
                identification_context=malicious_context,
                ctx=mock_context
            )
            
            # Should handle malicious input without executing scripts
            assert "success" in result
            if "username" in result:
                # Should not contain unescaped script tags
                assert "<script>" not in result["username"]
                assert "javascript:" not in result["username"]
    
    @pytest.mark.asyncio
    async def test_profile_management_path_traversal_prevention(self, mock_context):
        """Test path traversal prevention in profile management."""
        malicious_user_identities = [
            "../../../etc/passwd",
            "..\\..\\..\\windows\\system32\\config\\sam",
            "/etc/shadow",
            "C:\\Windows\\System32\\config\\SAM"
        ]
        
        for malicious_identity in malicious_user_identities:
            result = await km_manage_user_profiles(
                operation="list",
                user_identity=malicious_identity,
                ctx=mock_context
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
    async def test_privacy_level_enforcement(self, mock_context):
        """Test privacy level enforcement across tools."""
        # Test that minimal privacy level restricts data access
        minimal_result = await km_identify_user(
            identification_context={"username": "admin"},
            privacy_level="minimal",
            include_preferences=True,
            load_behavioral_data=True,
            ctx=mock_context
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
    async def test_authentication_performance(self, mock_context):
        """Test authentication performance requirements."""
        start_time = datetime.now(UTC)
        
        result = await km_authenticate_user(
            username="admin",
            authentication_method="password",
            password="SecureAdmin123!",
            ctx=mock_context
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
    async def test_identification_performance(self, mock_context):
        """Test identification performance requirements."""
        start_time = datetime.now(UTC)
        
        result = await km_identify_user(
            identification_context={"username": "admin"},
            load_behavioral_data=True,
            ctx=mock_context
        )
        
        end_time = datetime.now(UTC)
        execution_time = (end_time - start_time).total_seconds() * 1000
        
        # Should complete within reasonable time
        assert execution_time < 500
        
        if result["success"]:
            assert "processing_time_ms" in result
    
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_context):
        """Test concurrent user identity operations."""
        # Create multiple concurrent authentication requests
        tasks = []
        for i in range(5):
            task = km_authenticate_user(
                username="admin",
                authentication_method="session",
                ctx=mock_context
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
    async def test_complete_identity_workflow(self, mock_context, sample_user_data):
        """Test complete user identity workflow."""
        # Step 1: Authenticate user
        auth_result = await km_authenticate_user(
            username=sample_user_data["username"],
            authentication_method="password",
            password=sample_user_data["password"],
            ctx=mock_context
        )
        
        if not auth_result["success"]:
            # If auth fails, user might not exist - continue with admin
            username = "admin"
        else:
            username = auth_result["username"]
        
        # Step 2: Identify user and get profile
        identify_result = await km_identify_user(
            identification_context={"username": username},
            include_preferences=True,
            ctx=mock_context
        )
        
        assert identify_result["success"] == True
        user_profile_id = identify_result["user_profile_id"]
        
        # Step 3: Personalize automation
        personalize_result = await km_personalize_automation(
            user_identity=username,
            automation_context="workflow",
            adaptation_level="moderate",
            ctx=mock_context
        )
        
        assert personalize_result["success"] == True
        
        # Step 4: Analyze behavior
        behavior_result = await km_analyze_user_behavior(
            user_identity=username,
            analysis_period="week",
            ctx=mock_context
        )
        
        assert behavior_result["success"] == True
        
        # Step 5: Update preferences
        update_result = await km_manage_user_profiles(
            operation="update",
            user_identity=username,
            preferences={
                "personalization": {"automation_style": "advanced"}
            },
            ctx=mock_context
        )
        
        # Update may succeed or fail, but should handle gracefully
        assert "success" in update_result
    
    @pytest.mark.asyncio
    async def test_multi_user_context_workflow(self, mock_context):
        """Test multi-user context management workflow."""
        # Switch from admin to test user
        switch_result = await km_switch_user_context(
            target_user="testuser",
            current_user="admin",
            preserve_session=True,
            load_preferences=True,
            ctx=mock_context
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
                ctx=mock_context
            )
            
            assert "success" in switch_back
    
    @pytest.mark.asyncio
    async def test_privacy_workflow(self, mock_context):
        """Test privacy-focused workflow."""
        # Identify user with enhanced privacy
        identify_result = await km_identify_user(
            identification_context={"username": "admin"},
            privacy_level="enhanced",
            load_behavioral_data=True,
            ctx=mock_context
        )
        
        if identify_result["success"]:
            # Analyze behavior with privacy preservation
            behavior_result = await km_analyze_user_behavior(
                user_identity="admin",
                privacy_preserving=True,
                generate_insights=True,
                ctx=mock_context
            )
            
            assert behavior_result["success"] == True
            assert behavior_result["privacy"]["privacy_preserving"] == True
            
            # Manage profile with compliance
            manage_result = await km_manage_user_profiles(
                operation="list",
                user_identity="admin",
                compliance_mode=True,
                audit_logging=True,
                ctx=mock_context
            )
            
            assert manage_result["compliance"]["gdpr_compliant"] == True


# Property-based tests for robust validation
@pytest.mark.property
class TestUserIdentityToolsProperties:
    """Property-based tests for user identity tools."""
    
    @pytest.mark.asyncio
    async def test_authentication_input_validation_properties(self, mock_context):
        """Property: Authentication should validate all inputs safely."""
        import string
        import random
        
        # Generate random but safe test inputs
        test_cases = []
        for _ in range(10):
            username = ''.join(random.choices(string.ascii_letters + string.digits, k=random.randint(1, 50)))
            password = ''.join(random.choices(string.ascii_letters + string.digits + "!@#$%", k=random.randint(8, 100)))
            test_cases.append((username, password))
        
        for username, password in test_cases:
            result = await km_authenticate_user(
                username=username,
                authentication_method="password",
                password=password,
                ctx=mock_context
            )
            
            # Property: All results should have success field
            assert "success" in result
            assert isinstance(result["success"], bool)
            
            # Property: Processing time should always be reported
            assert "processing_time_ms" in result
            assert isinstance(result["processing_time_ms"], (int, float))
            assert result["processing_time_ms"] >= 0
    
    @pytest.mark.asyncio
    async def test_identification_context_properties(self, mock_context):
        """Property: Identification should handle various context formats."""
        test_contexts = [
            {"username": "admin"},
            {"user_id": "admin"},
            {"email": "admin@example.com"},
            {"username": "admin", "source": "test"},
            {"username": "admin", "timestamp": datetime.now(UTC).isoformat()}
        ]
        
        for context in test_contexts:
            result = await km_identify_user(
                identification_context=context,
                ctx=mock_context
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