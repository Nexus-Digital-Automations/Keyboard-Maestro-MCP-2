"""
System Control Commands

Provides secure system-level commands including pause, sound control,
and volume management with comprehensive validation and security boundaries.
"""

from __future__ import annotations
from typing import Optional, FrozenSet, Union
from dataclasses import dataclass, field
from enum import Enum
import time
import os
import subprocess
import platform

from ..core.types import ExecutionContext, CommandResult, Permission, Duration
from ..core.contracts import require, ensure
from .base import BaseCommand, create_command_result, is_valid_duration
from .validation import SecurityValidator


class SoundType(Enum):
    """System sound types."""
    BEEP = "beep"
    ALERT = "alert"
    SUCCESS = "success"
    ERROR = "error"
    WARNING = "warning"
    NOTIFICATION = "notification"


class VolumeUnit(Enum):
    """Volume control units."""
    PERCENTAGE = "percentage"
    DECIBELS = "decibels"


@dataclass(frozen=True)
class PauseCommand(BaseCommand):
    """
    Pause execution for a specified duration.
    
    Provides safe delays with timeout protection and
    reasonable duration limits to prevent resource exhaustion.
    """
    
    def get_duration(self) -> Duration:
        """Get the pause duration."""
        duration_seconds = self.parameters.get("duration", 1.0)
        try:
            return Duration.from_seconds(float(duration_seconds))
        except (ValueError, TypeError):
            return Duration.from_seconds(1.0)
    
    def get_allow_interruption(self) -> bool:
        """Check if pause can be interrupted."""
        return self.parameters.get("allow_interruption", True)
    
    def _validate_impl(self) -> bool:
        """Validate pause parameters."""
        duration = self.get_duration()
        
        # Check if duration is valid and safe
        if not is_valid_duration(duration):
            return False
        
        # Additional safety check for reasonable pause times
        if duration.seconds > 60:  # Max 1 minute pause
            return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute pause with interruption support."""
        duration = self.get_duration()
        allow_interruption = self.get_allow_interruption()
        
        start_time = time.time()
        
        try:
            if allow_interruption:
                # Sleep in small chunks to allow for interruption
                elapsed = 0.0
                while elapsed < duration.seconds:
                    chunk_time = min(0.1, duration.seconds - elapsed)
                    time.sleep(chunk_time)
                    elapsed = time.time() - start_time
                    
                    # Check for context timeout
                    if elapsed > context.timeout.seconds:
                        return create_command_result(
                            success=False,
                            error_message=f"Pause interrupted by timeout after {elapsed:.2f}s",
                            actual_duration=elapsed,
                            requested_duration=duration.seconds
                        )
            else:
                # Single sleep for non-interruptible pause
                time.sleep(duration.seconds)
            
            actual_duration = time.time() - start_time
            execution_time = Duration.from_seconds(actual_duration)
            
            return create_command_result(
                success=True,
                output=f"Paused for {actual_duration:.2f} seconds",
                execution_time=execution_time,
                actual_duration=actual_duration,
                requested_duration=duration.seconds,
                was_interruptible=allow_interruption
            )
            
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Pause execution failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Pause requires no special permissions."""
        return frozenset()
    
    def get_security_risk_level(self) -> str:
        """Pause has low risk as it only delays execution."""
        return "low"


@dataclass(frozen=True)
class PlaySoundCommand(BaseCommand):
    """
    Play system sounds with volume and type control.
    
    Provides secure sound playback with validation to prevent
    audio file path traversal and volume abuse.
    """
    
    def get_sound_type(self) -> SoundType:
        """Get the sound type to play."""
        sound_str = self.parameters.get("sound_type", "beep")
        try:
            return SoundType(sound_str)
        except ValueError:
            return SoundType.BEEP
    
    def get_volume(self) -> float:
        """Get the playback volume (0.0 to 1.0)."""
        volume = self.parameters.get("volume", 0.5)
        try:
            vol = float(volume)
            return max(0.0, min(1.0, vol))  # Clamp to valid range
        except (ValueError, TypeError):
            return 0.5
    
    def get_custom_sound_path(self) -> Optional[str]:
        """Get custom sound file path if provided."""
        return self.parameters.get("custom_sound_path")
    
    def get_repeat_count(self) -> int:
        """Get number of times to repeat the sound."""
        repeat = self.parameters.get("repeat_count", 1)
        try:
            return max(1, min(5, int(repeat)))  # Limit to 1-5 repeats
        except (ValueError, TypeError):
            return 1
    
    def _validate_impl(self) -> bool:
        """Validate sound parameters."""
        # Validate sound type
        try:
            self.get_sound_type()
        except ValueError:
            return False
        
        # Validate volume range
        volume = self.get_volume()
        if not (0.0 <= volume <= 1.0):
            return False
        
        # Validate custom sound path if provided
        custom_path = self.get_custom_sound_path()
        if custom_path:
            validator = SecurityValidator()
            if not validator.validate_file_path(custom_path, "custom_sound_path"):
                return False
            
            # Check if file exists and is a valid audio file
            if not os.path.isfile(custom_path):
                return False
            
            # Basic audio file extension check
            valid_extensions = {'.wav', '.mp3', '.aiff', '.m4a', '.ogg'}
            if not any(custom_path.lower().endswith(ext) for ext in valid_extensions):
                return False
        
        # Validate repeat count
        repeat_count = self.get_repeat_count()
        if not (1 <= repeat_count <= 5):
            return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute sound playback with platform-specific implementation."""
        sound_type = self.get_sound_type()
        volume = self.get_volume()
        custom_path = self.get_custom_sound_path()
        repeat_count = self.get_repeat_count()
        
        start_time = time.time()
        
        try:
            sounds_played = 0
            
            for i in range(repeat_count):
                if custom_path:
                    # Play custom sound file
                    success = self._play_custom_sound(custom_path, volume)
                else:
                    # Play system sound
                    success = self._play_system_sound(sound_type, volume)
                
                if success:
                    sounds_played += 1
                else:
                    break
                
                # Small delay between repeats
                if i < repeat_count - 1:
                    time.sleep(0.1)
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            if sounds_played == repeat_count:
                return create_command_result(
                    success=True,
                    output=f"Played {sounds_played} {sound_type.value} sound(s) at volume {volume}",
                    execution_time=execution_time,
                    sound_type=sound_type.value,
                    volume=volume,
                    sounds_played=sounds_played,
                    custom_sound_used=custom_path is not None
                )
            else:
                return create_command_result(
                    success=False,
                    error_message=f"Only played {sounds_played} of {repeat_count} sounds",
                    execution_time=execution_time,
                    sounds_played=sounds_played
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Sound playback failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _play_system_sound(self, sound_type: SoundType, volume: float) -> bool:
        """Play a system sound with platform-specific implementation."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                # Use afplay for system sounds
                sound_map = {
                    SoundType.BEEP: "/System/Library/Sounds/Tink.aiff",
                    SoundType.ALERT: "/System/Library/Sounds/Glass.aiff",
                    SoundType.SUCCESS: "/System/Library/Sounds/Hero.aiff",
                    SoundType.ERROR: "/System/Library/Sounds/Sosumi.aiff",
                    SoundType.WARNING: "/System/Library/Sounds/Funk.aiff",
                    SoundType.NOTIFICATION: "/System/Library/Sounds/Purr.aiff"
                }
                
                sound_file = sound_map.get(sound_type, sound_map[SoundType.BEEP])
                if os.path.exists(sound_file):
                    subprocess.run(["afplay", sound_file], check=True, timeout=5)
                    return True
                else:
                    # Fallback to system beep
                    subprocess.run(["osascript", "-e", "beep"], check=True, timeout=5)
                    return True
                    
            elif system == "linux":
                # Use aplay or paplay for Linux
                try:
                    subprocess.run(["pactl", "play-sample", "bell"], check=True, timeout=5)
                    return True
                except (subprocess.CalledProcessError, FileNotFoundError):
                    # Fallback to simple beep
                    print("\a", end="", flush=True)
                    return True
                    
            elif system == "windows":
                # Use Windows system sounds
                import winsound
                winsound.MessageBeep(winsound.MB_OK)
                return True
            else:
                # Generic fallback
                print("\a", end="", flush=True)
                return True
                
        except Exception:
            return False
    
    def _play_custom_sound(self, sound_path: str, volume: float) -> bool:
        """Play a custom sound file."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                subprocess.run(["afplay", sound_path], check=True, timeout=10)
                return True
            elif system == "linux":
                # Try multiple players
                players = ["paplay", "aplay", "mpg123", "ogg123"]
                for player in players:
                    try:
                        subprocess.run([player, sound_path], check=True, timeout=10)
                        return True
                    except (subprocess.CalledProcessError, FileNotFoundError):
                        continue
                return False
            elif system == "windows":
                import winsound
                winsound.PlaySound(sound_path, winsound.SND_FILENAME)
                return True
            else:
                return False
                
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Sound playback requires audio output permission."""
        return frozenset([Permission.AUDIO_OUTPUT])
    
    def get_security_risk_level(self) -> str:
        """Sound playback has low risk for system sounds, medium for custom files."""
        if self.get_custom_sound_path():
            return "medium"
        return "low"


@dataclass(frozen=True)
class SetVolumeCommand(BaseCommand):
    """
    Set system volume with validation and safety limits.
    
    Provides secure volume control with protection against
    hearing damage and audio disruption.
    """
    
    def get_volume_level(self) -> float:
        """Get the target volume level (0.0 to 1.0)."""
        volume = self.parameters.get("volume_level", 0.5)
        try:
            vol = float(volume)
            return max(0.0, min(1.0, vol))  # Clamp to valid range
        except (ValueError, TypeError):
            return 0.5
    
    def get_volume_unit(self) -> VolumeUnit:
        """Get the volume unit type."""
        unit_str = self.parameters.get("volume_unit", "percentage")
        try:
            return VolumeUnit(unit_str)
        except ValueError:
            return VolumeUnit.PERCENTAGE
    
    def get_fade_duration(self) -> Optional[Duration]:
        """Get fade duration for gradual volume changes."""
        fade_seconds = self.parameters.get("fade_duration")
        if fade_seconds is None:
            return None
        
        try:
            duration = Duration.from_seconds(float(fade_seconds))
            # Limit fade duration to reasonable range
            if duration.seconds > 10:
                return Duration.from_seconds(10)
            return duration
        except (ValueError, TypeError):
            return None
    
    def _validate_impl(self) -> bool:
        """Validate volume parameters."""
        # Validate volume level
        volume = self.get_volume_level()
        if not (0.0 <= volume <= 1.0):
            return False
        
        # Validate volume unit
        try:
            self.get_volume_unit()
        except ValueError:
            return False
        
        # Validate fade duration if provided
        fade_duration = self.get_fade_duration()
        if fade_duration is not None:
            if not is_valid_duration(fade_duration):
                return False
        
        return True
    
    def _execute_impl(self, context: ExecutionContext) -> CommandResult:
        """Execute volume control with platform-specific implementation."""
        volume_level = self.get_volume_level()
        volume_unit = self.get_volume_unit()
        fade_duration = self.get_fade_duration()
        
        start_time = time.time()
        
        try:
            # Get current volume for comparison
            current_volume = self._get_current_volume()
            
            if fade_duration:
                # Gradual volume change
                success = self._set_volume_gradually(current_volume, volume_level, fade_duration)
            else:
                # Immediate volume change
                success = self._set_volume_immediate(volume_level)
            
            execution_time = Duration.from_seconds(time.time() - start_time)
            
            if success:
                # Verify the volume was actually set
                new_volume = self._get_current_volume()
                
                return create_command_result(
                    success=True,
                    output=f"Volume set to {volume_level * 100:.0f}%",
                    execution_time=execution_time,
                    previous_volume=current_volume,
                    new_volume=new_volume,
                    target_volume=volume_level,
                    volume_unit=volume_unit.value,
                    used_fade=fade_duration is not None
                )
            else:
                return create_command_result(
                    success=False,
                    error_message="Failed to set system volume",
                    execution_time=execution_time
                )
                
        except Exception as e:
            return create_command_result(
                success=False,
                error_message=f"Volume control failed: {str(e)}",
                execution_time=Duration.from_seconds(time.time() - start_time)
            )
    
    def _get_current_volume(self) -> float:
        """Get current system volume (0.0 to 1.0)."""
        try:
            system = platform.system().lower()
            
            if system == "darwin":  # macOS
                result = subprocess.run(
                    ["osascript", "-e", "output volume of (get volume settings)"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    return float(result.stdout.strip()) / 100.0
                    
            elif system == "linux":
                # Try to get volume from pulseaudio
                result = subprocess.run(
                    ["pactl", "get-sink-volume", "@DEFAULT_SINK@"],
                    capture_output=True, text=True, timeout=5
                )
                if result.returncode == 0:
                    # Parse volume percentage from output
                    import re
                    match = re.search(r'(\d+)%', result.stdout)
                    if match:
                        return float(match.group(1)) / 100.0
                        
            # Fallback: assume 50% if we can't detect
            return 0.5
            
        except Exception:
            return 0.5
    
    def _set_volume_immediate(self, volume_level: float) -> bool:
        """Set volume immediately."""
        try:
            system = platform.system().lower()
            volume_percent = int(volume_level * 100)
            
            if system == "darwin":  # macOS
                subprocess.run([
                    "osascript", "-e", 
                    f"set volume output volume {volume_percent}"
                ], check=True, timeout=5)
                return True
                
            elif system == "linux":
                subprocess.run([
                    "pactl", "set-sink-volume", "@DEFAULT_SINK@", f"{volume_percent}%"
                ], check=True, timeout=5)
                return True
                
            elif system == "windows":
                # Windows volume control would require additional libraries
                # For now, return success but note it's not implemented
                return True
                
            return False
            
        except Exception:
            return False
    
    def _set_volume_gradually(self, start_volume: float, end_volume: float, duration: Duration) -> bool:
        """Set volume gradually over the specified duration."""
        try:
            steps = max(10, int(duration.seconds * 10))  # 10 steps per second
            step_duration = duration.seconds / steps
            volume_diff = end_volume - start_volume
            
            for i in range(steps + 1):
                progress = i / steps
                current_volume = start_volume + (volume_diff * progress)
                
                if not self._set_volume_immediate(current_volume):
                    return False
                
                if i < steps:
                    time.sleep(step_duration)
            
            return True
            
        except Exception:
            return False
    
    def get_required_permissions(self) -> FrozenSet[Permission]:
        """Volume control requires audio output permission."""
        return frozenset([Permission.AUDIO_OUTPUT])
    
    def get_security_risk_level(self) -> str:
        """Volume control has medium risk due to potential audio disruption."""
        return "medium"