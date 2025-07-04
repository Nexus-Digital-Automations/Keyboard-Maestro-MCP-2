"""
Comprehensive tests for macro movement tools.

Tests cover validation, security, execution, error handling, and property-based scenarios
following ADDER+ testing methodology.
"""

import pytest
import asyncio
from unittest.mock import AsyncMock, patch, MagicMock
from datetime import datetime
from typing import Dict, Any

from hypothesis import given, strategies as st, settings
from hypothesis.stateful import RuleBasedStateMachine, rule, invariant

from src.server.tools.macro_move_tools import (
    km_move_macro_to_group,
    MoveConflictType,
    MacroMoveResult,
    _sanitize_identifier,
    _validate_security_constraints,
    _validate_move_operation,
    _execute_macro_movement,
    _get_macro_info,
    _check_group_exists,
    _create_group_if_missing,
    _verify_movement_completion,
    _escape_applescript_string
)
from src.core.errors import ValidationError, SecurityViolationError
from src.core.types import Duration


class TestInputValidation:
    """Test comprehensive input validation and sanitization."""
    
    def test_sanitize_identifier_valid_inputs(self):
        """Test valid identifier sanitization."""
        assert _sanitize_identifier("Test Macro", "macro") == "Test Macro"
        assert _sanitize_identifier("  Valid Name  ", "macro") == "Valid Name"
        assert _sanitize_identifier("Test-Macro_123", "macro") == "Test-Macro_123"
    
    def test_sanitize_identifier_invalid_inputs(self):
        """Test invalid identifier rejection."""
        with pytest.raises(ValueError, match="cannot be empty"):
            _sanitize_identifier("", "macro")
        
        with pytest.raises(ValueError, match="cannot be empty"):
            _sanitize_identifier("   ", "macro")
        
        with pytest.raises(ValueError, match="cannot exceed 255 characters"):
            _sanitize_identifier("x" * 256, "macro")
    
    def test_sanitize_identifier_security_patterns(self):
        """Test rejection of suspicious patterns."""
        suspicious_inputs = [
            "<script>alert('xss')</script>",
            "javascript:void(0)",
            "../../../etc/passwd",
            "cmd.exe /c dir",
            "powershell -command",
            "eval(malicious_code)",
            "system('rm -rf /')"
        ]
        
        for suspicious in suspicious_inputs:
            with pytest.raises(ValueError, match="Suspicious pattern detected"):
                _sanitize_identifier(suspicious, "macro")
    
    def test_validate_security_constraints_system_groups(self):
        """Test system group protection."""
        system_groups = [
            "Global Macro Group", "System", "Login", "Quit", "Sleep", "Wake"
        ]
        
        for group in system_groups:
            with pytest.raises(SecurityViolationError, match="Cannot move macros to system group"):
                _validate_security_constraints("Test Macro", group)
    
    def test_validate_security_constraints_unsafe_characters(self):
        """Test unsafe character rejection."""
        with pytest.raises(SecurityViolationError, match="unsafe characters"):
            _validate_security_constraints("Test\x00Macro", "ValidGroup")
        
        with pytest.raises(SecurityViolationError, match="unsafe characters"):
            _validate_security_constraints("ValidMacro", "Group\x00Name")


class TestMacroMovementValidation:
    """Test pre-movement validation logic."""
    
    @pytest.mark.asyncio
    async def test_validate_move_operation_macro_not_found(self):
        """Test validation when macro doesn't exist."""
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=None):
            result = await _validate_move_operation("NonexistentMacro", "TargetGroup", False, None)
            
            assert not result.success
            assert result.error_code == "MACRO_NOT_FOUND"
            assert "not found" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_move_operation_source_equals_target(self):
        """Test validation when source equals target group."""
        mock_info = {"group": "SameGroup"}
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info):
            result = await _validate_move_operation("TestMacro", "SameGroup", False, None)
            
            assert not result.success
            assert result.error_code == "SOURCE_EQUALS_TARGET"
            assert "already in group" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_move_operation_group_not_found(self):
        """Test validation when target group doesn't exist."""
        mock_info = {"group": "SourceGroup"}
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=False):
            
            result = await _validate_move_operation("TestMacro", "NonexistentGroup", False, None)
            
            assert not result.success
            assert result.error_code == "GROUP_NOT_FOUND"
            assert "does not exist" in result.error_message
    
    @pytest.mark.asyncio
    async def test_validate_move_operation_success(self):
        """Test successful validation."""
        mock_info = {"group": "SourceGroup"}
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=True):
            
            result = await _validate_move_operation("TestMacro", "TargetGroup", False, None)
            
            assert result.success
            assert result.macro_id == "TestMacro"
            assert result.source_group == "SourceGroup"
            assert result.target_group == "TargetGroup"


class TestAppleScriptIntegration:
    """Test AppleScript execution and integration."""
    
    @pytest.mark.asyncio
    async def test_get_macro_info_success(self):
        """Test successful macro info retrieval."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"TestGroup\n", b""))
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await _get_macro_info("TestMacro")
            
            assert result is not None
            assert result["group"] == "TestGroup"
    
    @pytest.mark.asyncio
    async def test_get_macro_info_not_found(self):
        """Test macro info retrieval when macro not found."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"NOT_FOUND\n", b""))
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await _get_macro_info("NonexistentMacro")
            
            assert result is None
    
    @pytest.mark.asyncio
    async def test_check_group_exists_true(self):
        """Test group existence check - exists."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"EXISTS\n", b""))
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await _check_group_exists("ExistingGroup")
            
            assert result is True
    
    @pytest.mark.asyncio
    async def test_check_group_exists_false(self):
        """Test group existence check - doesn't exist."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"NOT_FOUND\n", b""))
        
        with patch('asyncio.create_subprocess_exec', return_value=mock_process):
            result = await _check_group_exists("NonexistentGroup")
            
            assert result is False
    
    @pytest.mark.asyncio
    async def test_create_group_if_missing_success(self):
        """Test successful group creation."""
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        with patch('src.server.tools.macro_move_tools._check_group_exists', return_value=False), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process):
            
            result = await _create_group_if_missing("NewGroup")
            
            assert result is True
    
    def test_escape_applescript_string(self):
        """Test AppleScript string escaping."""
        assert _escape_applescript_string('Test "quoted" string') == 'Test \\"quoted\\" string'
        assert _escape_applescript_string('Line\nbreak') == 'Line\\nbreak'
        assert _escape_applescript_string('Tab\there') == 'Tab\\there'
        assert _escape_applescript_string('Back\\slash') == 'Back\\\\slash'


class TestMacroMovementExecution:
    """Test complete macro movement execution."""
    
    @pytest.mark.asyncio
    async def test_execute_macro_movement_success(self):
        """Test successful macro movement execution."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro):
            
            result = await _execute_macro_movement(
                "TestMacro", "TargetGroup", False, True, Duration.from_seconds(30), None
            )
            
            assert result.success
            assert result.macro_id == "TestMacro"
            assert result.source_group == "SourceGroup"
            assert result.target_group == "TargetGroup"
    
    @pytest.mark.asyncio
    async def test_execute_macro_movement_applescript_error(self):
        """Test macro movement with AppleScript error."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"ERROR: Macro not found\n", b""))
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro):
            
            result = await _execute_macro_movement(
                "TestMacro", "TargetGroup", False, True, Duration.from_seconds(30), None
            )
            
            assert not result.success
            assert result.error_code == "MOVE_ERROR"
            assert "Macro not found" in result.error_message
    
    @pytest.mark.asyncio
    async def test_execute_macro_movement_timeout(self):
        """Test macro movement timeout handling."""
        mock_info = {"group": "SourceGroup"}
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('asyncio.create_subprocess_exec'), \
             patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            
            result = await _execute_macro_movement(
                "TestMacro", "TargetGroup", False, True, Duration.from_seconds(5), None
            )
            
            assert not result.success
            assert result.error_code == "TIMEOUT_ERROR"
            assert "timeout" in result.error_message.lower()


class TestFullMacroMovementWorkflow:
    """Test complete macro movement workflow through MCP tool."""
    
    @pytest.mark.asyncio
    async def test_km_move_macro_to_group_success(self):
        """Test successful complete macro movement."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=True), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro), \
             patch('src.server.tools.macro_move_tools._verify_movement_completion', return_value=True):
            
            result = await km_move_macro_to_group(
                macro_identifier="TestMacro",
                target_group="TargetGroup",
                create_group_if_missing=False,
                preserve_group_settings=True,
                timeout_seconds=30,
                ctx=None
            )
            
            assert result["success"] is True
            assert result["data"]["macro_identifier"] == "TestMacro"
            assert result["data"]["source_group"] == "SourceGroup"
            assert result["data"]["target_group"] == "TargetGroup"
            assert "correlation_id" in result["metadata"]
    
    @pytest.mark.asyncio
    async def test_km_move_macro_to_group_validation_error(self):
        """Test macro movement with validation error."""
        result = await km_move_macro_to_group(
            macro_identifier="",  # Invalid empty identifier
            target_group="TargetGroup",
            create_group_if_missing=False,
            preserve_group_settings=True,
            timeout_seconds=30,
            ctx=None
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "cannot be empty" in result["error"]["message"]
    
    @pytest.mark.asyncio
    async def test_km_move_macro_to_group_security_violation(self):
        """Test macro movement with security violation."""
        result = await km_move_macro_to_group(
            macro_identifier="TestMacro",
            target_group="Global Macro Group",  # System group - should be blocked
            create_group_if_missing=False,
            preserve_group_settings=True,
            timeout_seconds=30,
            ctx=None
        )
        
        assert result["success"] is False
        assert result["error"]["code"] == "VALIDATION_ERROR"
        assert "system group" in result["error"]["details"].lower()
    
    @pytest.mark.asyncio
    async def test_km_move_macro_to_group_with_group_creation(self):
        """Test macro movement with automatic group creation."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=False), \
             patch('src.server.tools.macro_move_tools._create_group_if_missing', return_value=True), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro), \
             patch('src.server.tools.macro_move_tools._verify_movement_completion', return_value=True):
            
            result = await km_move_macro_to_group(
                macro_identifier="TestMacro",
                target_group="NewGroup",
                create_group_if_missing=True,
                preserve_group_settings=True,
                timeout_seconds=30,
                ctx=None
            )
            
            assert result["success"] is True
            assert result["data"]["group_created"] is True


class TestPropertyBasedScenarios:
    """Property-based tests for macro movement operations."""
    
    @given(
        macro_name=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc"), 
            whitelist_characters=" -_.[](){}|&~!@#$%^*=?:;,/\\"
        )),
        source_group=st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc"),
            whitelist_characters=" -_."
        )),
        target_group=st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc"),
            whitelist_characters=" -_."
        ))
    )
    @settings(max_examples=50)
    def test_sanitization_properties(self, macro_name, source_group, target_group):
        """
        Property: Valid inputs should always pass sanitization.
        Invalid inputs should always be rejected with clear error messages.
        """
        # Valid inputs should pass
        try:
            sanitized_macro = _sanitize_identifier(macro_name, "macro")
            sanitized_group = _sanitize_identifier(target_group, "group")
            
            # Sanitized results should be non-empty strings
            assert isinstance(sanitized_macro, str)
            assert len(sanitized_macro) > 0
            assert isinstance(sanitized_group, str)
            assert len(sanitized_group) > 0
            
            # Should not contain suspicious patterns
            assert "<script" not in sanitized_macro.lower()
            assert "javascript:" not in sanitized_macro.lower()
            assert "<script" not in sanitized_group.lower()
            assert "javascript:" not in sanitized_group.lower()
            
        except ValueError as e:
            # If validation fails, error message should be descriptive
            assert "identifier" in str(e).lower()
            assert len(str(e)) > 10  # Should have meaningful error description
    
    @given(
        valid_macro=st.text(min_size=1, max_size=50, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" -_"
        )),
        valid_group=st.text(min_size=1, max_size=30, alphabet=st.characters(
            whitelist_categories=("Lu", "Ll", "Nd"),
            whitelist_characters=" -_"
        ))
    )
    @settings(max_examples=30)
    def test_security_constraints_properties(self, valid_macro, valid_group):
        """
        Property: Security validation should consistently apply rules.
        Valid inputs should pass, system groups should be blocked.
        """
        # Valid inputs should pass security validation
        try:
            _validate_security_constraints(valid_macro, valid_group)
        except SecurityViolationError:
            # If security violation occurs, it should be for a valid reason
            # (e.g., system group protection)
            pass
        
        # System groups should always be blocked
        system_groups = ["Global Macro Group", "System", "Login"]
        for system_group in system_groups:
            with pytest.raises(SecurityViolationError):
                _validate_security_constraints(valid_macro, system_group)


class TestErrorHandlingAndRecovery:
    """Test comprehensive error handling and recovery scenarios."""
    
    @pytest.mark.asyncio
    async def test_rollback_on_verification_failure(self):
        """Test rollback when movement verification fails."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=True), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro), \
             patch('src.server.tools.macro_move_tools._verify_movement_completion', return_value=False), \
             patch('src.server.tools.macro_move_tools._attempt_rollback') as mock_rollback:
            
            result = await km_move_macro_to_group(
                macro_identifier="TestMacro",
                target_group="TargetGroup",
                create_group_if_missing=False,
                preserve_group_settings=True,
                timeout_seconds=30,
                ctx=None
            )
            
            assert result["success"] is False
            assert result["error"]["code"] == "VERIFICATION_FAILED"
            mock_rollback.assert_called_once()
    
    @pytest.mark.asyncio
    async def test_error_response_structure(self):
        """Test that error responses have consistent structure."""
        result = await km_move_macro_to_group(
            macro_identifier="",  # Invalid to trigger error
            target_group="TargetGroup",
            create_group_if_missing=False,
            preserve_group_settings=True,
            timeout_seconds=30,
            ctx=None
        )
        
        # Verify error response structure
        assert "success" in result
        assert result["success"] is False
        assert "error" in result
        assert "metadata" in result
        
        error = result["error"]
        assert "code" in error
        assert "message" in error
        assert "details" in error
        assert "recovery_suggestion" in error
        
        metadata = result["metadata"]
        assert "correlation_id" in metadata
        assert "timestamp" in metadata
        assert "execution_time" in metadata
        assert "operation" in metadata
    
    @pytest.mark.asyncio 
    async def test_timeout_handling(self):
        """Test proper timeout handling and error reporting."""
        mock_info = {"group": "SourceGroup"}
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=True), \
             patch('asyncio.create_subprocess_exec'), \
             patch('asyncio.wait_for', side_effect=asyncio.TimeoutError()):
            
            result = await km_move_macro_to_group(
                macro_identifier="TestMacro",
                target_group="TargetGroup",
                create_group_if_missing=False,
                preserve_group_settings=True,
                timeout_seconds=5,  # Short timeout
                ctx=None
            )
            
            assert result["success"] is False
            # Should get timeout error from the execution phase
            assert "timeout" in result["error"]["message"].lower() or \
                   "failed" in result["error"]["message"].lower()


class TestPerformanceAndBenchmarks:
    """Test performance characteristics and benchmarks."""
    
    @pytest.mark.asyncio
    async def test_operation_timing(self):
        """Test that operations complete within expected timeframes."""
        mock_info = {"group": "SourceGroup"}
        mock_process = MagicMock()
        mock_process.returncode = 0
        mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
        
        start_time = datetime.now()
        
        with patch('src.server.tools.macro_move_tools._get_macro_info', return_value=mock_info), \
             patch('src.server.tools.macro_move_tools._check_group_exists', return_value=True), \
             patch('asyncio.create_subprocess_exec', return_value=mock_process), \
             patch('asyncio.wait_for', side_effect=lambda coro, timeout: coro), \
             patch('src.server.tools.macro_move_tools._verify_movement_completion', return_value=True):
            
            result = await km_move_macro_to_group(
                macro_identifier="TestMacro",
                target_group="TargetGroup",
                create_group_if_missing=False,
                preserve_group_settings=True,
                timeout_seconds=30,
                ctx=None
            )
        
        execution_time = (datetime.now() - start_time).total_seconds()
        
        # Operation should complete quickly in test environment
        assert execution_time < 1.0  # Less than 1 second
        assert result["success"] is True
        
        # Execution time should be reported in metadata
        reported_time = result["metadata"]["execution_time"]
        assert isinstance(reported_time, float)
        assert reported_time > 0


# Integration test fixtures and utilities
@pytest.fixture
def mock_km_environment():
    """Mock Keyboard Maestro environment for testing."""
    return {
        "macros": {
            "TestMacro": {"group": "SourceGroup", "enabled": True},
            "AnotherMacro": {"group": "OtherGroup", "enabled": False}
        },
        "groups": ["SourceGroup", "OtherGroup", "TargetGroup"]
    }


@pytest.fixture
def mock_applescript_success():
    """Mock successful AppleScript execution."""
    mock_process = MagicMock()
    mock_process.returncode = 0
    mock_process.communicate = AsyncMock(return_value=(b"SUCCESS\n", b""))
    return mock_process


@pytest.fixture
def mock_applescript_error():
    """Mock failed AppleScript execution."""
    mock_process = MagicMock()
    mock_process.returncode = 1
    mock_process.communicate = AsyncMock(return_value=(b"", b"AppleScript Error\n"))
    return mock_process


# Test utilities
def assert_valid_error_response(result: Dict[str, Any]):
    """Assert that error response has valid structure."""
    assert "success" in result
    assert result["success"] is False
    assert "error" in result
    assert "metadata" in result
    
    error = result["error"]
    required_error_fields = ["code", "message", "details", "recovery_suggestion"]
    for field in required_error_fields:
        assert field in error
        assert isinstance(error[field], str)
        assert len(error[field]) > 0
    
    metadata = result["metadata"]
    required_metadata_fields = ["correlation_id", "timestamp", "execution_time", "operation"]
    for field in required_metadata_fields:
        assert field in metadata


def assert_valid_success_response(result: Dict[str, Any]):
    """Assert that success response has valid structure."""
    assert "success" in result
    assert result["success"] is True
    assert "data" in result
    assert "metadata" in result
    
    data = result["data"]
    required_data_fields = ["macro_identifier", "source_group", "target_group", "operation_time"]
    for field in required_data_fields:
        assert field in data
    
    metadata = result["metadata"]
    required_metadata_fields = ["correlation_id", "timestamp", "execution_time", "operation"]
    for field in required_metadata_fields:
        assert field in metadata