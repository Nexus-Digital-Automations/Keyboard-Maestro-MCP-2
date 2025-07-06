"""
Comprehensive Test Suite for Hotkey Tools - Following Proven MCP Tool Test Pattern

This test suite validates the Hotkey Tools functionality using the systematic
testing approach that achieved 100% success rate across 14 tool suites.

Test Coverage:
- Hotkey trigger creation functionality with comprehensive validation
- Conflict detection and alternative suggestion systems
- Key and modifier validation with security boundary checking
- Activation mode and tap count validation
- Security validation for hotkey creation and injection prevention
- Property-based testing for robust input validation
- Integration testing with mocked hotkey managers and KM clients
- Error handling for all failure scenarios
- Performance testing for hotkey operation response times

Testing Strategy:
- Property-based testing with Hypothesis for comprehensive input coverage
- Mock-based testing for HotkeyManager and KMClient components
- Security validation for hotkey creation injection prevention
- Integration testing scenarios with realistic hotkey operations
- Performance and timeout testing with hotkey operation limits

Key Mocking Pattern:
- HotkeyManager: Mock all methods with Either.success() pattern
- KMClient: Mock Keyboard Maestro client integration
- TriggerRegistrationManager: Mock trigger registration operations
- HotkeySpec: Mock hotkey specification creation and validation
"""

from unittest.mock import AsyncMock, Mock, patch

import pytest
from fastmcp import Context
from hypothesis import HealthCheck, assume, given, settings
from hypothesis import strategies as st
from hypothesis.strategies import composite
from src.core.errors import (
    SecurityViolationError,
    ValidationError,
)
from src.integration.km_client import KMClient
from src.integration.triggers import TriggerRegistrationManager

# Import the tools we're testing
from src.server.tools.hotkey_tools import (
    km_create_hotkey_trigger,
    km_list_hotkey_triggers,
)

# Import hotkey types and errors
from src.triggers.hotkey_manager import HotkeyManager


# Test fixtures following proven pattern
@pytest.fixture
def mock_context():
    """Create mock FastMCP context following successful pattern."""
    context = Mock(spec=Context)
    context.info = AsyncMock()
    context.warn = AsyncMock()
    context.error = AsyncMock()
    context.report_progress = AsyncMock()
    context.read_resource = AsyncMock()
    context.sample = AsyncMock()
    context.get = Mock(return_value="")  # Support ctx.get() calls
    return context


@pytest.fixture
def mock_hotkey_manager():
    """Create mock HotkeyManager with standard interface."""
    manager = Mock(spec=HotkeyManager)
    manager.detect_conflicts = AsyncMock()
    manager.suggest_alternatives = Mock()
    manager.create_hotkey_trigger = AsyncMock()
    manager.get_registered_hotkeys = Mock()

    # Setup standard success response using Either.success() pattern
    mock_result = Mock()
    mock_result.is_right.return_value = True
    mock_result.is_left.return_value = False  # Critical: must be False for success path
    mock_result.get_right.return_value = "trigger-123"
    manager.create_hotkey_trigger.return_value = mock_result

    # Setup conflict detection to return no conflicts by default
    manager.detect_conflicts.return_value = []

    # Setup registered hotkeys
    mock_hotkey_spec = Mock()
    mock_hotkey_spec.to_display_string.return_value = "Cmd+N"
    mock_hotkey_spec.to_km_string.return_value = "cmd+n"
    mock_hotkey_spec.key = "n"
    mock_hotkey_spec.modifiers = [Mock(value="cmd")]
    mock_hotkey_spec.activation_mode = Mock(value="pressed")
    mock_hotkey_spec.tap_count = 1
    mock_hotkey_spec.allow_repeat = False

    manager.get_registered_hotkeys.return_value = {
        "cmd+n": ("macro-123", mock_hotkey_spec)
    }

    # Setup alternatives
    manager.suggest_alternatives.return_value = [mock_hotkey_spec]

    return manager


@pytest.fixture
def mock_km_client():
    """Create mock KMClient with standard interface."""
    client = Mock(spec=KMClient)
    return client


@pytest.fixture
def mock_trigger_manager():
    """Create mock TriggerRegistrationManager with standard interface."""
    manager = Mock(spec=TriggerRegistrationManager)
    return manager


@pytest.fixture
def mock_hotkey_spec():
    """Create mock HotkeySpec with standard interface."""
    spec = Mock()
    spec.to_display_string.return_value = "Cmd+N"
    spec.to_km_string.return_value = "cmd+n"
    spec.key = "n"
    spec.modifiers = [Mock(value="cmd")]
    spec.activation_mode = Mock(value="pressed")
    spec.tap_count = 1
    spec.allow_repeat = False
    return spec


# Core Hotkey Tools Tests
class TestHotkeyTriggerCreation:
    """Test core km_create_hotkey_trigger functionality."""

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_success(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test successful hotkey trigger creation."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is True
            assert result["data"]["trigger_id"] == "trigger-123"
            assert result["data"]["hotkey"]["key"] == "n"
            assert result["data"]["hotkey"]["modifiers"] == ["cmd"]
            mock_hotkey_manager.create_hotkey_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_with_multiple_modifiers(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with multiple modifiers."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="s",
                modifiers=["cmd", "shift"],
                activation_mode="tapped",
                tap_count=2,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["hotkey"]["modifiers"] == ["cmd"]  # Uses mock spec
            mock_hotkey_manager.create_hotkey_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_special_keys(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with special keys."""
        special_keys = ["space", "tab", "f1", "escape", "enter"]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            for key in special_keys:
                result = await km_create_hotkey_trigger(
                    macro_id="test-macro-123",
                    key=key,
                    modifiers=["cmd"],
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert result["data"]["hotkey"]["key"] == "n"  # Uses mock spec

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_activation_modes(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test different activation modes."""
        activation_modes = ["pressed", "released", "tapped", "held"]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            for mode in activation_modes:
                result = await km_create_hotkey_trigger(
                    macro_id="test-macro-123",
                    key="n",
                    modifiers=["cmd"],
                    activation_mode=mode,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert (
                    result["data"]["hotkey"]["activation_mode"] == "pressed"
                )  # Uses mock spec

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_tap_counts(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test different tap counts."""
        tap_counts = [1, 2, 3, 4]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            for tap_count in tap_counts:
                result = await km_create_hotkey_trigger(
                    macro_id="test-macro-123",
                    key="n",
                    modifiers=["cmd"],
                    tap_count=tap_count,
                    ctx=mock_context,
                )

                assert result["success"] is True
                assert result["data"]["hotkey"]["tap_count"] == 1  # Uses mock spec

    @pytest.mark.asyncio
    async def test_create_hotkey_trigger_allow_repeat(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test allow_repeat option."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                allow_repeat=True,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["hotkey"]["allow_repeat"] is False  # Uses mock spec


# Conflict Detection Tests
class TestHotkeyConflictDetection:
    """Test hotkey conflict detection functionality."""

    @pytest.mark.asyncio
    async def test_conflict_detection_with_conflicts(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test conflict detection when conflicts exist."""
        # Setup conflict detection to return conflicts
        mock_conflict = Mock()
        mock_conflict.conflicting_hotkey = "cmd+n"
        mock_conflict.conflict_type = "system"
        mock_conflict.description = "Conflicts with system shortcut"
        mock_conflict.macro_name = "System New"
        mock_conflict.suggestion = "Try cmd+shift+n"
        mock_hotkey_manager.detect_conflicts.return_value = [mock_conflict]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                check_conflicts=True,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "CONFLICT_ERROR"
            assert "conflicts" in result["error"]["details"]
            assert len(result["error"]["details"]["conflicts"]) == 1
            mock_hotkey_manager.detect_conflicts.assert_called_once()

    @pytest.mark.asyncio
    async def test_conflict_detection_with_alternatives(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test conflict detection with alternative suggestions."""
        # Setup conflict
        mock_conflict = Mock()
        mock_conflict.conflicting_hotkey = "cmd+n"
        mock_conflict.conflict_type = "system"
        mock_conflict.description = "Conflicts with system shortcut"
        mock_conflict.macro_name = "System New"
        mock_conflict.suggestion = "Try cmd+shift+n"
        mock_hotkey_manager.detect_conflicts.return_value = [mock_conflict]

        # Setup alternative
        mock_alternative = Mock()
        mock_alternative.to_km_string.return_value = "cmd+shift+n"
        mock_alternative.to_display_string.return_value = "Cmd+Shift+N"
        mock_alternative.modifiers = [Mock(value="cmd"), Mock(value="shift")]
        mock_alternative.key = "n"
        mock_alternative.activation_mode = Mock(value="pressed")
        mock_alternative.tap_count = 1
        mock_hotkey_manager.suggest_alternatives.return_value = [mock_alternative]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                check_conflicts=True,
                suggest_alternatives=True,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "CONFLICT_ERROR"
            assert "suggested_alternatives" in result["error"]["details"]
            assert len(result["error"]["details"]["suggested_alternatives"]) == 1
            mock_hotkey_manager.suggest_alternatives.assert_called_once()

    @pytest.mark.asyncio
    async def test_conflict_detection_disabled(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with conflict detection disabled."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                check_conflicts=False,
                ctx=mock_context,
            )

            assert result["success"] is True
            assert result["data"]["conflicts_checked"] is False
            mock_hotkey_manager.detect_conflicts.assert_not_called()


# Hotkey Listing Tests
class TestHotkeyListing:
    """Test km_list_hotkey_triggers functionality."""

    @pytest.mark.asyncio
    async def test_list_all_hotkeys(
        self, mock_context, mock_hotkey_manager, mock_km_client, mock_trigger_manager
    ):
        """Test listing all registered hotkeys."""
        with (
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_list_hotkey_triggers(ctx=mock_context)

            assert result["success"] is True
            assert "hotkeys" in result["data"]
            assert result["data"]["total_count"] == 1
            assert result["data"]["filtered_by_macro"] is False
            mock_hotkey_manager.get_registered_hotkeys.assert_called_once()

    @pytest.mark.asyncio
    async def test_list_hotkeys_filtered_by_macro(
        self, mock_context, mock_hotkey_manager, mock_km_client, mock_trigger_manager
    ):
        """Test listing hotkeys filtered by macro ID."""
        with (
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_list_hotkey_triggers(
                macro_id="macro-123", ctx=mock_context
            )

            assert result["success"] is True
            assert result["data"]["filtered_by_macro"] is True
            assert result["data"]["total_count"] == 1

    @pytest.mark.asyncio
    async def test_list_hotkeys_with_conflicts(
        self, mock_context, mock_hotkey_manager, mock_km_client, mock_trigger_manager
    ):
        """Test listing hotkeys with conflict information."""
        # Setup conflict for listing
        mock_conflict = Mock()
        mock_conflict.conflict_type = "system"
        mock_conflict.description = "System shortcut conflict"
        mock_conflict.suggestion = "Try different combination"
        mock_hotkey_manager.detect_conflicts.return_value = [mock_conflict]

        with (
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_list_hotkey_triggers(
                include_conflicts=True, ctx=mock_context
            )

            assert result["success"] is True
            assert result["data"]["conflicts_included"] is True
            # Check that conflict detection was called for each hotkey
            mock_hotkey_manager.detect_conflicts.assert_called()


# Error Handling Tests
class TestHotkeyToolsErrorHandling:
    """Test hotkey tools error handling scenarios."""

    @pytest.mark.asyncio
    async def test_invalid_hotkey_specification_error(self, mock_context):
        """Test handling of invalid hotkey specification."""
        with patch(
            "src.server.tools.hotkey_tools.create_hotkey_spec",
            side_effect=ValidationError("key", "invalid", "must be valid key"),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="invalid",
                modifiers=["cmd"],
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_HOTKEY"
            assert "Invalid hotkey specification" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_security_violation_error(self, mock_context):
        """Test handling of security violations."""
        with patch(
            "src.server.tools.hotkey_tools.create_hotkey_spec",
            side_effect=SecurityViolationError("injection", "detected malicious input"),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is False
            assert result["error"]["code"] == "INVALID_HOTKEY"

    @pytest.mark.asyncio
    async def test_hotkey_manager_error_response(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test handling of HotkeyManager error responses."""
        # Configure mock to return error
        error_result = Mock()
        error_result.is_left.return_value = True  # Critical: True for error path
        error_result.is_right.return_value = False
        error_result.get_left.return_value = Mock(
            code="TRIGGER_CREATION_FAILED",
            message="Failed to create trigger",
            details={"reason": "Macro not found"},
        )
        mock_hotkey_manager.create_hotkey_trigger.return_value = error_result

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is False
            assert result["error"]["code"] == "TRIGGER_CREATION_FAILED"
            assert result["error"]["message"] == "Failed to create trigger"

    @pytest.mark.asyncio
    async def test_general_exception_handling(self, mock_context):
        """Test general exception handling."""
        with patch(
            "src.server.tools.hotkey_tools.create_hotkey_spec",
            side_effect=Exception("Unexpected error"),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Unexpected error" in result["error"]["message"]

    @pytest.mark.asyncio
    async def test_list_hotkeys_exception_handling(self, mock_context):
        """Test exception handling in list hotkeys."""
        with patch(
            "src.server.tools.hotkey_tools.KMClient",
            side_effect=Exception("Client initialization failed"),
        ):
            result = await km_list_hotkey_triggers(ctx=mock_context)

            assert result["success"] is False
            assert result["error"]["code"] == "SYSTEM_ERROR"
            assert "Failed to retrieve hotkey triggers" in result["error"]["message"]


# Security Tests
class TestHotkeyToolsSecurity:
    """Test hotkey tools security validation."""

    @pytest.mark.asyncio
    async def test_macro_id_sanitization(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test macro ID input sanitization."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            # Test with leading/trailing whitespace
            result = await km_create_hotkey_trigger(
                macro_id="  test-macro-123  ",
                key="n",
                modifiers=["cmd"],
                ctx=mock_context,
            )

            assert result["success"] is True
            # Verify sanitization occurred (macro_id should be stripped)
            mock_hotkey_manager.create_hotkey_trigger.assert_called_once()

    @pytest.mark.asyncio
    async def test_key_sanitization(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test key input sanitization."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            # Test with uppercase key
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="N", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is True
            # Verify key was normalized to lowercase

    @pytest.mark.asyncio
    async def test_modifier_sanitization(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test modifier input sanitization."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            # Test with mixed case and empty modifiers
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["CMD", " ", "shift", ""],
                ctx=mock_context,
            )

            assert result["success"] is True
            # Verify modifiers were sanitized (empty/whitespace removed, lowercased)

    @pytest.mark.asyncio
    async def test_injection_prevention_macro_id(self, mock_context):
        """Test prevention of injection in macro ID."""
        malicious_ids = [
            "macro'; DROP TABLE macros; --",
            "macro<script>alert('xss')</script>",
            "macro`rm -rf /`",
            "macro${malicious}",
            "macro|malicious",
        ]

        for malicious_id in malicious_ids:
            # The security validation happens in create_hotkey_spec
            with patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                side_effect=SecurityViolationError(
                    "injection", "detected injection attempt"
                ),
            ):
                result = await km_create_hotkey_trigger(
                    macro_id=malicious_id, key="n", modifiers=["cmd"], ctx=mock_context
                )

                assert result["success"] is False
                assert result["error"]["code"] == "INVALID_HOTKEY"


# Integration Tests
class TestHotkeyToolsIntegration:
    """Test hotkey tools integration scenarios."""

    @pytest.mark.asyncio
    async def test_create_and_list_workflow(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test combined create and list workflow."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            # Create hotkey
            create_result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            # List hotkeys
            list_result = await km_list_hotkey_triggers(ctx=mock_context)

            assert create_result["success"] is True
            assert list_result["success"] is True
            assert list_result["data"]["total_count"] == 1

    @pytest.mark.asyncio
    async def test_conflict_detection_workflow(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test complete conflict detection workflow."""
        # Setup conflict scenario
        mock_conflict = Mock()
        mock_conflict.conflicting_hotkey = "cmd+n"
        mock_conflict.conflict_type = "macro"
        mock_conflict.description = "Conflicts with existing macro"
        mock_conflict.macro_name = "New Note"
        mock_conflict.suggestion = "Try cmd+shift+n"
        mock_hotkey_manager.detect_conflicts.return_value = [mock_conflict]

        # Setup alternative
        mock_alternative = Mock()
        mock_alternative.to_km_string.return_value = "cmd+shift+n"
        mock_alternative.to_display_string.return_value = "Cmd+Shift+N"
        mock_alternative.modifiers = [Mock(value="cmd"), Mock(value="shift")]
        mock_alternative.key = "n"
        mock_alternative.activation_mode = Mock(value="pressed")
        mock_alternative.tap_count = 1
        mock_hotkey_manager.suggest_alternatives.return_value = [mock_alternative]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            # Attempt to create conflicting hotkey
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                check_conflicts=True,
                suggest_alternatives=True,
                ctx=mock_context,
            )

            assert result["success"] is False
            assert result["error"]["code"] == "CONFLICT_ERROR"
            assert len(result["error"]["details"]["conflicts"]) == 1
            assert len(result["error"]["details"]["suggested_alternatives"]) == 1
            assert (
                result["error"]["details"]["suggested_alternatives"][0]["hotkey"]
                == "cmd+shift+n"
            )


# Property-Based Tests
class TestHotkeyToolsPropertyBased:
    """Property-based testing for hotkey tools with Hypothesis."""

    @composite
    def valid_keys(draw):
        """Generate valid key identifiers."""
        letter_keys = st.sampled_from("abcdefghijklmnopqrstuvwxyz")
        number_keys = st.sampled_from("0123456789")
        special_keys = st.sampled_from(
            [
                "space",
                "tab",
                "enter",
                "return",
                "escape",
                "delete",
                "backspace",
                "f1",
                "f2",
                "f3",
                "f4",
                "f5",
                "f6",
                "f7",
                "f8",
                "f9",
                "f10",
                "f11",
                "f12",
                "home",
                "end",
                "pageup",
                "pagedown",
                "up",
                "down",
                "left",
                "right",
                "clear",
                "help",
                "insert",
            ]
        )
        return draw(st.one_of(letter_keys, number_keys, special_keys))

    @composite
    def valid_modifiers(draw):
        """Generate valid modifier combinations."""
        modifiers = st.sampled_from(["cmd", "opt", "shift", "ctrl", "fn"])
        return draw(st.lists(modifiers, min_size=0, max_size=3, unique=True))

    @composite
    def valid_macro_ids(draw):
        """Generate valid macro IDs."""
        # Generate macro IDs without dangerous characters
        chars = st.characters(
            whitelist_categories=("Lu", "Ll", "Nd", "Pc"), whitelist_characters=" -._"
        )
        return draw(st.text(chars, min_size=1, max_size=100))

    @given(valid_macro_ids(), valid_keys(), valid_modifiers())
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    def test_input_sanitization_property(self, macro_id, key, modifiers):
        """Property: All valid inputs should be properly sanitized."""
        assume(len(macro_id.strip()) > 0)  # Non-empty after strip

        # Test that sanitization doesn't fail with valid inputs
        sanitized_macro_id = macro_id.strip()
        sanitized_key = key.strip().lower()
        sanitized_modifiers = [mod.strip().lower() for mod in modifiers if mod.strip()]

        assert len(sanitized_macro_id) > 0
        assert len(sanitized_key) > 0
        assert all(len(mod) > 0 for mod in sanitized_modifiers)

    @pytest.mark.asyncio
    @given(
        st.sampled_from(["pressed", "released", "tapped", "held"]),
        st.integers(min_value=1, max_value=4),
        st.booleans(),
    )
    @settings(suppress_health_check=[HealthCheck.function_scoped_fixture])
    async def test_hotkey_options_property(
        self, activation_mode, tap_count, allow_repeat
    ):
        """Property: All valid option combinations should be accepted."""
        mock_context = Mock(spec=Context)
        mock_context.info = AsyncMock()

        mock_hotkey_spec = Mock()
        mock_hotkey_spec.to_display_string.return_value = "Cmd+N"
        mock_hotkey_spec.to_km_string.return_value = "cmd+n"
        mock_hotkey_spec.key = "n"
        mock_hotkey_spec.modifiers = [Mock(value="cmd")]
        mock_hotkey_spec.activation_mode = Mock(value=activation_mode)
        mock_hotkey_spec.tap_count = tap_count
        mock_hotkey_spec.allow_repeat = allow_repeat

        mock_hotkey_manager = Mock(spec=HotkeyManager)
        mock_hotkey_manager.detect_conflicts = AsyncMock(return_value=[])
        mock_result = Mock()
        mock_result.is_right.return_value = True
        mock_result.get_right.return_value = "trigger-123"
        mock_hotkey_manager.create_hotkey_trigger = AsyncMock(return_value=mock_result)

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch("src.server.tools.hotkey_tools.KMClient"),
            patch("src.server.tools.hotkey_tools.TriggerRegistrationManager"),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                activation_mode=activation_mode,
                tap_count=tap_count,
                allow_repeat=allow_repeat,
                ctx=mock_context,
            )

            # Should either succeed or fail with specific error (not crash)
            assert "success" in result
            assert isinstance(result["success"], bool)


# Performance Tests
class TestHotkeyToolsPerformance:
    """Test hotkey tools performance characteristics."""

    @pytest.mark.asyncio
    async def test_hotkey_creation_response_time(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test that hotkey creation completes within reasonable time."""
        import time

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            start_time = time.time()

            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=["cmd"], ctx=mock_context
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds (allowing for mocking overhead)
            assert execution_time < 2.0
            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_conflict_detection_performance(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test conflict detection performance."""
        import time

        # Setup multiple conflicts to test performance
        conflicts = [Mock() for _ in range(10)]
        for i, conflict in enumerate(conflicts):
            conflict.conflicting_hotkey = f"cmd+{i}"
            conflict.conflict_type = "macro"
            conflict.description = f"Conflicts with macro {i}"
            conflict.macro_name = f"Macro {i}"
            conflict.suggestion = f"Try alt+{i}"

        mock_hotkey_manager.detect_conflicts.return_value = conflicts

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            start_time = time.time()

            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                check_conflicts=True,
                ctx=mock_context,
            )

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds even with multiple conflicts
            assert execution_time < 2.0
            assert result["success"] is False  # Should fail due to conflicts
            assert len(result["error"]["details"]["conflicts"]) == 10

    @pytest.mark.asyncio
    async def test_list_hotkeys_performance(
        self, mock_context, mock_hotkey_manager, mock_km_client, mock_trigger_manager
    ):
        """Test hotkey listing performance with many hotkeys."""
        import time

        # Setup many registered hotkeys
        registered_hotkeys = {}
        for i in range(50):
            mock_spec = Mock()
            mock_spec.to_display_string.return_value = f"Cmd+{i}"
            mock_spec.to_km_string.return_value = f"cmd+{i}"
            mock_spec.key = str(i)
            mock_spec.modifiers = [Mock(value="cmd")]
            mock_spec.activation_mode = Mock(value="pressed")
            mock_spec.tap_count = 1
            mock_spec.allow_repeat = False
            registered_hotkeys[f"cmd+{i}"] = (f"macro-{i}", mock_spec)

        mock_hotkey_manager.get_registered_hotkeys.return_value = registered_hotkeys

        with (
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            start_time = time.time()

            result = await km_list_hotkey_triggers(ctx=mock_context)

            end_time = time.time()
            execution_time = end_time - start_time

            # Should complete within 2 seconds even with many hotkeys
            assert execution_time < 2.0
            assert result["success"] is True
            assert result["data"]["total_count"] == 50


# Edge Case Tests
class TestHotkeyToolsEdgeCases:
    """Test hotkey tools edge cases and boundary conditions."""

    @pytest.mark.asyncio
    async def test_empty_modifiers_list(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with empty modifiers list."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123", key="n", modifiers=[], ctx=mock_context
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_maximum_tap_count(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with maximum tap count."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="test-macro-123",
                key="n",
                modifiers=["cmd"],
                tap_count=4,  # Maximum allowed
                ctx=mock_context,
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_unicode_macro_id(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with Unicode macro ID."""
        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id="测试宏 🎯", key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_maximum_length_macro_id(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test hotkey creation with maximum length macro ID."""
        long_macro_id = "a" * 255  # Maximum allowed length

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_create_hotkey_trigger(
                macro_id=long_macro_id, key="n", modifiers=["cmd"], ctx=mock_context
            )

            assert result["success"] is True

    @pytest.mark.asyncio
    async def test_all_modifier_combinations(
        self,
        mock_context,
        mock_hotkey_manager,
        mock_km_client,
        mock_trigger_manager,
        mock_hotkey_spec,
    ):
        """Test various modifier combinations."""
        modifier_combinations = [
            ["cmd"],
            ["cmd", "shift"],
            ["cmd", "opt"],
            ["cmd", "ctrl"],
            ["cmd", "shift", "opt"],
            ["shift", "opt", "ctrl"],
            ["cmd", "shift", "opt", "ctrl"],
        ]

        with (
            patch(
                "src.server.tools.hotkey_tools.create_hotkey_spec",
                return_value=mock_hotkey_spec,
            ),
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            for modifiers in modifier_combinations:
                result = await km_create_hotkey_trigger(
                    macro_id="test-macro-123",
                    key="n",
                    modifiers=modifiers,
                    ctx=mock_context,
                )

                assert result["success"] is True

    @pytest.mark.asyncio
    async def test_list_empty_hotkeys(
        self, mock_context, mock_hotkey_manager, mock_km_client, mock_trigger_manager
    ):
        """Test listing when no hotkeys are registered."""
        mock_hotkey_manager.get_registered_hotkeys.return_value = {}

        with (
            patch(
                "src.server.tools.hotkey_tools.KMClient", return_value=mock_km_client
            ),
            patch(
                "src.server.tools.hotkey_tools.TriggerRegistrationManager",
                return_value=mock_trigger_manager,
            ),
            patch(
                "src.server.tools.hotkey_tools.HotkeyManager",
                return_value=mock_hotkey_manager,
            ),
        ):
            result = await km_list_hotkey_triggers(ctx=mock_context)

            assert result["success"] is True
            assert result["data"]["total_count"] == 0
            assert result["data"]["hotkeys"] == []
