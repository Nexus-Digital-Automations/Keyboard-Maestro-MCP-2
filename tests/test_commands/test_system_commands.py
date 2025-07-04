"""
Tests for system control commands.

Tests pause, sound, and volume control commands with security validation
and proper contract enforcement.
"""

import pytest
import time
from unittest.mock import patch, MagicMock

from src.core.types import (
    CommandId, CommandParameters, ExecutionContext, Permission, Duration
)
from src.commands.system import PauseCommand, PlaySoundCommand, SetVolumeCommand
from src.commands.base import create_command_result


class TestPauseCommand:
    """Test pause command functionality."""
    
    def test_pause_command_creation(self):
        """Test basic pause command creation."""
        params = CommandParameters({"duration": 0.1})
        cmd = PauseCommand(CommandId("test"), params)
        
        assert cmd.get_duration().seconds == 0.1
        assert cmd.get_allow_interruption() is True
    
    def test_pause_validation_valid(self):
        """Test pause command validation with valid parameters."""
        params = CommandParameters({"duration": 1.0, "allow_interruption": True})
        cmd = PauseCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_pause_validation_invalid_duration(self):
        """Test pause command validation with invalid duration."""
        # Too long duration
        params = CommandParameters({"duration": 70.0})
        cmd = PauseCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
        
        # Negative duration - this should be caught by Duration constructor
        # but let's test with a very small duration that should fail
        params = CommandParameters({"duration": 0.0})
        cmd = PauseCommand(CommandId("test"), params)
        
        assert cmd.validate() is False
    
    def test_pause_execution(self):
        """Test pause command execution."""
        params = CommandParameters({"duration": 0.1, "allow_interruption": True})
        cmd = PauseCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset(),
            timeout=Duration.from_seconds(10)
        )
        
        start = time.time()
        result = cmd.execute(context)
        elapsed = time.time() - start
        
        assert result.success is True
        assert elapsed >= 0.08  # Allow some tolerance
        assert "Paused for" in result.output
    
    def test_pause_permissions(self):
        """Test pause command permission requirements."""
        params = CommandParameters({"duration": 1.0})
        cmd = PauseCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert len(permissions) == 0  # Pause requires no special permissions
    
    def test_pause_security_risk(self):
        """Test pause command security risk level."""
        params = CommandParameters({"duration": 1.0})
        cmd = PauseCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "low"


class TestPlaySoundCommand:
    """Test sound playback command functionality."""
    
    def test_sound_command_creation(self):
        """Test basic sound command creation."""
        params = CommandParameters({
            "sound_type": "beep",
            "volume": 0.5,
            "repeat_count": 1
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        assert cmd.get_sound_type().value == "beep"
        assert cmd.get_volume() == 0.5
        assert cmd.get_repeat_count() == 1
    
    def test_sound_validation_valid(self):
        """Test sound command validation with valid parameters."""
        params = CommandParameters({
            "sound_type": "alert",
            "volume": 0.7,
            "repeat_count": 2
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_sound_validation_invalid_volume(self):
        """Test sound command validation with invalid volume."""
        params = CommandParameters({
            "sound_type": "beep",
            "volume": 1.5  # Invalid volume > 1.0
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        # Volume should be clamped to valid range
        assert cmd.get_volume() == 1.0
        assert cmd.validate() is True
    
    def test_sound_validation_invalid_repeat_count(self):
        """Test sound command validation with invalid repeat count."""
        params = CommandParameters({
            "sound_type": "beep",
            "repeat_count": 10  # Too many repeats
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        # Repeat count should be clamped
        assert cmd.get_repeat_count() == 5
        assert cmd.validate() is True
    
    @patch('subprocess.run')
    def test_sound_execution_system_sound(self, mock_subprocess):
        """Test sound command execution with system sound."""
        mock_subprocess.return_value.returncode = 0
        
        params = CommandParameters({
            "sound_type": "beep",
            "volume": 0.5,
            "repeat_count": 1
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.AUDIO_OUTPUT]),
            timeout=Duration.from_seconds(10)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Played 1 beep sound" in result.output
    
    def test_sound_permissions(self):
        """Test sound command permission requirements."""
        params = CommandParameters({"sound_type": "beep"})
        cmd = PlaySoundCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.AUDIO_OUTPUT in permissions
    
    def test_sound_security_risk(self):
        """Test sound command security risk level."""
        # System sound
        params = CommandParameters({"sound_type": "beep"})
        cmd = PlaySoundCommand(CommandId("test"), params)
        assert cmd.get_security_risk_level() == "low"
        
        # Custom sound file
        params = CommandParameters({
            "sound_type": "beep",
            "custom_sound_path": "/tmp/test.wav"
        })
        cmd = PlaySoundCommand(CommandId("test"), params)
        assert cmd.get_security_risk_level() == "medium"


class TestSetVolumeCommand:
    """Test volume control command functionality."""
    
    def test_volume_command_creation(self):
        """Test basic volume command creation."""
        params = CommandParameters({
            "volume_level": 0.7,
            "volume_unit": "percentage"
        })
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        assert cmd.get_volume_level() == 0.7
        assert cmd.get_volume_unit().value == "percentage"
    
    def test_volume_validation_valid(self):
        """Test volume command validation with valid parameters."""
        params = CommandParameters({
            "volume_level": 0.8,
            "fade_duration": 2.0
        })
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        assert cmd.validate() is True
    
    def test_volume_validation_invalid_level(self):
        """Test volume command validation with invalid level."""
        params = CommandParameters({"volume_level": 1.5})
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        # Volume should be clamped to valid range
        assert cmd.get_volume_level() == 1.0
        assert cmd.validate() is True
    
    def test_volume_validation_invalid_fade_duration(self):
        """Test volume command validation with invalid fade duration."""
        params = CommandParameters({
            "volume_level": 0.5,
            "fade_duration": 15.0  # Too long
        })
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        # Fade duration should be clamped
        fade_duration = cmd.get_fade_duration()
        assert fade_duration is not None
        assert fade_duration.seconds == 10.0
    
    @patch('subprocess.run')
    def test_volume_execution(self, mock_subprocess):
        """Test volume command execution."""
        mock_subprocess.return_value.returncode = 0
        mock_subprocess.return_value.stdout = "50"
        
        params = CommandParameters({"volume_level": 0.8})
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        context = ExecutionContext.create_test_context(
            permissions=frozenset([Permission.AUDIO_OUTPUT]),
            timeout=Duration.from_seconds(10)
        )
        
        result = cmd.execute(context)
        
        assert result.success is True
        assert "Volume set to 80%" in result.output
    
    def test_volume_permissions(self):
        """Test volume command permission requirements."""
        params = CommandParameters({"volume_level": 0.5})
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        permissions = cmd.get_required_permissions()
        assert Permission.AUDIO_OUTPUT in permissions
    
    def test_volume_security_risk(self):
        """Test volume command security risk level."""
        params = CommandParameters({"volume_level": 0.5})
        cmd = SetVolumeCommand(CommandId("test"), params)
        
        assert cmd.get_security_risk_level() == "medium"


if __name__ == "__main__":
    pytest.main([__file__])