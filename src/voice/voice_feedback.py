"""
Voice Feedback System - TASK_66 Phase 2 Core Voice Engine

Text-to-speech response system with natural voice synthesis, emotional tone,
and comprehensive audio output management for voice command feedback.

Architecture: Voice Synthesis + Audio Output + Emotional Processing + Accessibility Features
Performance: <500ms synthesis start, <2s audio generation, <100ms playback start
Security: Content validation, safe audio processing, protected voice data handling
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, UTC
from dataclasses import dataclass, field
from enum import Enum
import asyncio
import logging
import tempfile
import subprocess
from pathlib import Path

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.voice_architecture import (
    VoiceLanguage, SpeakerId, VoiceControlError,
    VoiceProfile
)

logger = logging.getLogger(__name__)


class VoiceTone(Enum):
    """Emotional tones for voice feedback."""
    NEUTRAL = "neutral"
    FRIENDLY = "friendly"
    PROFESSIONAL = "professional"
    ENCOURAGING = "encouraging"
    WARNING = "warning"
    ERROR = "error"
    SUCCESS = "success"
    EXCITED = "excited"
    CALM = "calm"
    URGENT = "urgent"


class VoiceGender(Enum):
    """Voice gender options for synthesis."""
    MALE = "male"
    FEMALE = "female"
    NEUTRAL = "neutral"


@dataclass(frozen=True)
class VoiceSettings:
    """Voice synthesis settings and configuration."""
    voice_id: str
    voice_name: str
    language: VoiceLanguage
    gender: VoiceGender
    speech_rate: float = 200.0  # Words per minute
    pitch: float = 1.0  # Pitch multiplier
    volume: float = 0.8  # Volume level (0.0-1.0)
    tone: VoiceTone = VoiceTone.NEUTRAL
    
    @require(lambda self: 50.0 <= self.speech_rate <= 400.0)
    @require(lambda self: 0.5 <= self.pitch <= 2.0)
    @require(lambda self: 0.0 <= self.volume <= 1.0)
    def __post_init__(self):
        pass
    
    def to_applescript_params(self) -> str:
        """Convert to AppleScript speech parameters."""
        return f'speaking rate {self.speech_rate} pitch {self.pitch * 50} modulation 50'
    
    def adjust_for_tone(self, tone: VoiceTone) -> 'VoiceSettings':
        """Create adjusted settings for specific tone."""
        adjustments = {
            VoiceTone.FRIENDLY: (1.1, 1.05, 0.8),      # Slightly faster, higher pitch
            VoiceTone.PROFESSIONAL: (1.0, 1.0, 0.8),   # Standard settings
            VoiceTone.ENCOURAGING: (1.2, 1.1, 0.9),    # Faster, higher, louder
            VoiceTone.WARNING: (0.8, 0.9, 0.9),        # Slower, lower, louder
            VoiceTone.ERROR: (0.7, 0.8, 1.0),          # Slow, low, loud
            VoiceTone.SUCCESS: (1.3, 1.2, 0.9),        # Fast, high, moderate
            VoiceTone.EXCITED: (1.4, 1.3, 1.0),        # Very fast, very high
            VoiceTone.CALM: (0.9, 0.95, 0.7),          # Slow, slightly low, quiet
            VoiceTone.URGENT: (1.5, 1.1, 1.0),         # Very fast, high, loud
        }
        
        if tone not in adjustments:
            return self
        
        rate_mult, pitch_mult, vol_mult = adjustments[tone]
        
        return VoiceSettings(
            voice_id=self.voice_id,
            voice_name=self.voice_name,
            language=self.language,
            gender=self.gender,
            speech_rate=min(400.0, max(50.0, self.speech_rate * rate_mult)),
            pitch=min(2.0, max(0.5, self.pitch * pitch_mult)),
            volume=min(1.0, max(0.0, self.volume * vol_mult)),
            tone=tone
        )


@dataclass
class SpeechRequest:
    """Text-to-speech synthesis request."""
    text: str
    voice_settings: VoiceSettings
    output_destination: str = "system"  # "system", "file", "both"
    output_file_path: Optional[str] = None
    interrupt_current: bool = False
    save_audio: bool = False
    ssml_enabled: bool = False
    
    @require(lambda self: len(self.text.strip()) > 0)
    @require(lambda self: len(self.text) <= 5000)  # Reasonable length limit
    def __post_init__(self):
        pass
    
    def prepare_text_for_speech(self) -> str:
        """Prepare text for speech synthesis with SSML if enabled."""
        if self.ssml_enabled:
            return self.text
        
        # Basic text preparation for natural speech
        prepared_text = self.text
        
        # Handle common abbreviations
        abbreviations = {
            "Dr.": "Doctor",
            "Mr.": "Mister", 
            "Mrs.": "Missus",
            "Ms.": "Miss",
            "Prof.": "Professor",
            "vs.": "versus",
            "etc.": "etcetera",
            "e.g.": "for example",
            "i.e.": "that is",
            "URL": "U R L",
            "API": "A P I",
            "UI": "U I",
            "CPU": "C P U",
            "GPU": "G P U",
            "RAM": "R A M",
            "SSD": "S S D",
            "USB": "U S B"
        }
        
        for abbrev, expansion in abbreviations.items():
            prepared_text = prepared_text.replace(abbrev, expansion)
        
        # Handle numbers for natural pronunciation
        import re
        
        # Handle years (e.g., "2023" -> "twenty twenty three")
        year_pattern = r'\b(19|20)\d{2}\b'
        years = re.findall(year_pattern, prepared_text)
        for year_match in re.finditer(year_pattern, prepared_text):
            year = year_match.group()
            # Simple year pronunciation logic
            if year.startswith('20'):
                if year.endswith('00'):
                    spoken = f"twenty {year[2:4]} hundred" if year[2:4] != '00' else "two thousand"
                else:
                    spoken = f"twenty {year[2:4]}"
            else:  # 19xx
                spoken = f"nineteen {year[2:4]}"
            prepared_text = prepared_text.replace(year, spoken, 1)
        
        # Handle phone numbers (basic format)
        phone_pattern = r'\b\d{3}-\d{3}-\d{4}\b'
        for phone_match in re.finditer(phone_pattern, prepared_text):
            phone = phone_match.group()
            # Convert to spoken format: "555-123-4567" -> "five five five, one two three, four five six seven"
            digits = phone.replace('-', '')
            spoken_digits = []
            for digit in digits:
                spoken_digits.append({
                    '0': 'zero', '1': 'one', '2': 'two', '3': 'three', '4': 'four',
                    '5': 'five', '6': 'six', '7': 'seven', '8': 'eight', '9': 'nine'
                }.get(digit, digit))
            
            spoken_phone = f"{' '.join(spoken_digits[:3])}, {' '.join(spoken_digits[3:6])}, {' '.join(spoken_digits[6:])}"
            prepared_text = prepared_text.replace(phone, spoken_phone, 1)
        
        return prepared_text


@dataclass
class SpeechResult:
    """Text-to-speech synthesis result."""
    text: str
    voice_used: str
    synthesis_time_ms: float
    audio_duration_seconds: float
    output_file: Optional[str] = None
    success: bool = True
    error_message: Optional[str] = None
    
    @require(lambda self: self.synthesis_time_ms >= 0.0)
    @require(lambda self: self.audio_duration_seconds >= 0.0)
    def __post_init__(self):
        pass
    
    def is_successful(self) -> bool:
        """Check if synthesis was successful."""
        return self.success and self.error_message is None


class VoiceFeedbackSystem:
    """
    Comprehensive voice feedback and text-to-speech system.
    
    Contracts:
        Preconditions:
            - Speech text must be validated for content and length
            - Voice settings must be within valid parameter ranges
            - Audio output destinations must be accessible and secure
        
        Postconditions:
            - Speech synthesis completes within performance constraints
            - Audio output is generated with requested quality and settings
            - Feedback timing is appropriate for voice command workflows
        
        Invariants:
            - Voice settings remain within safe parameter ranges
            - Audio files are properly managed and cleaned up
            - Speech content is validated for security and appropriateness
    """
    
    def __init__(self):
        self.available_voices: Dict[str, VoiceSettings] = {}
        self.current_playback: Optional[subprocess.Popen] = None
        self.synthesis_cache: Dict[str, SpeechResult] = {}
        self.feedback_stats = {
            "total_requests": 0,
            "successful_synthesis": 0,
            "failed_synthesis": 0,
            "average_synthesis_time": 0.0,
            "total_audio_duration": 0.0
        }
        
        # Initialize available system voices
        self._initialize_system_voices()
    
    def _initialize_system_voices(self):
        """Initialize available system voices."""
        # macOS system voices
        system_voices = [
            ("alex", "Alex", VoiceLanguage.ENGLISH_US, VoiceGender.MALE),
            ("samantha", "Samantha", VoiceLanguage.ENGLISH_US, VoiceGender.FEMALE),
            ("victoria", "Victoria", VoiceLanguage.ENGLISH_US, VoiceGender.FEMALE),
            ("daniel", "Daniel", VoiceLanguage.ENGLISH_UK, VoiceGender.MALE),
            ("karen", "Karen", VoiceLanguage.ENGLISH_AU, VoiceGender.FEMALE),
            ("jorge", "Jorge", VoiceLanguage.SPANISH_ES, VoiceGender.MALE),
            ("paulina", "Paulina", VoiceLanguage.SPANISH_MX, VoiceGender.FEMALE),
            ("thomas", "Thomas", VoiceLanguage.FRENCH_FR, VoiceGender.MALE),
            ("amelie", "Amelie", VoiceLanguage.FRENCH_FR, VoiceGender.FEMALE),
        ]
        
        for voice_id, name, language, gender in system_voices:
            self.available_voices[voice_id] = VoiceSettings(
                voice_id=voice_id,
                voice_name=name,
                language=language,
                gender=gender
            )
        
        logger.info(f"Initialized {len(self.available_voices)} system voices")
    
    @require(lambda self, request: len(request.text.strip()) > 0)
    @ensure(lambda self, result: result.is_success() or result.error_value)
    async def synthesize_speech(self, request: SpeechRequest) -> Either[VoiceControlError, SpeechResult]:
        """
        Synthesize text to speech with specified voice settings.
        
        Performance:
            - <500ms synthesis initialization
            - <2s audio generation for typical responses
            - <100ms playback start for system output
        """
        try:
            start_time = datetime.now(UTC)
            
            # Validate speech content
            validation_result = self._validate_speech_content(request.text)
            if validation_result.is_error():
                return validation_result
            
            # Check cache for identical request
            cache_key = self._generate_cache_key(request)
            if cache_key in self.synthesis_cache:
                cached_result = self.synthesis_cache[cache_key]
                logger.info(f"Using cached speech synthesis: {request.text[:50]}...")
                return Either.success(cached_result)
            
            # Prepare text for speech
            prepared_text = request.prepare_text_for_speech()
            
            # Stop current playback if interruption requested
            if request.interrupt_current and self.current_playback:
                await self._stop_current_playback()
            
            # Perform synthesis based on output destination
            if request.output_destination in ["system", "both"]:
                system_result = await self._synthesize_to_system(prepared_text, request.voice_settings)
                if system_result.is_error():
                    return system_result
            
            if request.output_destination in ["file", "both"] or request.save_audio:
                file_result = await self._synthesize_to_file(prepared_text, request)
                if file_result.is_error():
                    return file_result
                output_file = file_result.value
            else:
                output_file = None
            
            # Calculate synthesis time and estimate audio duration
            synthesis_time = (datetime.now(UTC) - start_time).total_seconds() * 1000
            audio_duration = self._estimate_audio_duration(prepared_text, request.voice_settings)
            
            # Create result
            result = SpeechResult(
                text=request.text,
                voice_used=request.voice_settings.voice_name,
                synthesis_time_ms=synthesis_time,
                audio_duration_seconds=audio_duration,
                output_file=output_file,
                success=True
            )
            
            # Cache result
            self.synthesis_cache[cache_key] = result
            
            # Update statistics
            self._update_synthesis_stats(result, True)
            
            logger.info(f"Speech synthesis successful: '{request.text[:50]}...' "
                       f"(voice: {request.voice_settings.voice_name}, "
                       f"time: {synthesis_time:.0f}ms, duration: {audio_duration:.1f}s)")
            
            return Either.success(result)
            
        except Exception as e:
            self._update_synthesis_stats(None, False)
            error_msg = f"Speech synthesis failed: {str(e)}"
            logger.error(error_msg)
            return Either.error(VoiceControlError(error_msg))
    
    async def _synthesize_to_system(
        self,
        text: str,
        voice_settings: VoiceSettings
    ) -> Either[VoiceControlError, None]:
        """Synthesize speech directly to system audio output."""
        try:
            # Build AppleScript for speech synthesis
            applescript = f'''
            set speechText to "{self._escape_applescript_text(text)}"
            set speechVoice to "{voice_settings.voice_id}"
            set speechRate to {voice_settings.speech_rate}
            set speechPitch to {voice_settings.pitch * 50}
            set speechVolume to {voice_settings.volume}
            
            say speechText using speechVoice speaking rate speechRate pitch speechPitch modulation 50
            '''
            
            # Execute AppleScript
            process = await asyncio.create_subprocess_shell(
                f'osascript -e \'{applescript}\'',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.current_playback = process
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "AppleScript execution failed"
                return Either.error(VoiceControlError(f"System speech synthesis failed: {error_msg}"))
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"System speech synthesis error: {str(e)}"))
        finally:
            self.current_playback = None
    
    async def _synthesize_to_file(
        self,
        text: str,
        request: SpeechRequest
    ) -> Either[VoiceControlError, str]:
        """Synthesize speech to audio file."""
        try:
            # Determine output file path
            if request.output_file_path:
                output_path = request.output_file_path
            else:
                # Create temporary file
                temp_dir = Path(tempfile.gettempdir()) / "voice_feedback"
                temp_dir.mkdir(exist_ok=True)
                timestamp = datetime.now(UTC).strftime("%Y%m%d_%H%M%S")
                output_path = str(temp_dir / f"speech_{timestamp}.aiff")
            
            # Build AppleScript for file synthesis
            applescript = f'''
            set speechText to "{self._escape_applescript_text(text)}"
            set speechVoice to "{request.voice_settings.voice_id}"
            set speechRate to {request.voice_settings.speech_rate}
            set speechPitch to {request.voice_settings.pitch * 50}
            set outputFile to POSIX file "{output_path}"
            
            say speechText using speechVoice speaking rate speechRate pitch speechPitch modulation 50 saving to outputFile
            '''
            
            # Execute AppleScript
            process = await asyncio.create_subprocess_shell(
                f'osascript -e \'{applescript}\'',
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            stdout, stderr = await process.communicate()
            
            if process.returncode != 0:
                error_msg = stderr.decode() if stderr else "AppleScript execution failed"
                return Either.error(VoiceControlError(f"File speech synthesis failed: {error_msg}"))
            
            # Verify file was created
            if not Path(output_path).exists():
                return Either.error(VoiceControlError("Speech file was not created"))
            
            return Either.success(output_path)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"File speech synthesis error: {str(e)}"))
    
    def _escape_applescript_text(self, text: str) -> str:
        """Escape text for AppleScript."""
        # Escape quotes and backslashes
        escaped = text.replace('\\', '\\\\').replace('"', '\\"')
        return escaped
    
    def _validate_speech_content(self, text: str) -> Either[VoiceControlError, None]:
        """Validate speech content for security and appropriateness."""
        try:
            # Check length
            if len(text) > 5000:
                return Either.error(VoiceControlError("Speech text too long"))
            
            if not text.strip():
                return Either.error(VoiceControlError("Speech text is empty"))
            
            # Check for sensitive patterns
            import re
            sensitive_patterns = [
                r'(?i)(password|secret|token|api[_\s]*key)[\s:=]+[^\s]+',
                r'(?i)(credit[_\s]*card|ssn|social[_\s]*security)',
                r'<script[^>]*>.*?</script>',
                r'javascript:',
            ]
            
            for pattern in sensitive_patterns:
                if re.search(pattern, text):
                    return Either.error(VoiceControlError("Speech content contains sensitive information"))
            
            return Either.success(None)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"Speech content validation failed: {str(e)}"))
    
    def _estimate_audio_duration(self, text: str, voice_settings: VoiceSettings) -> float:
        """Estimate audio duration in seconds."""
        # Rough estimation based on speech rate and text length
        word_count = len(text.split())
        words_per_minute = voice_settings.speech_rate
        duration_minutes = word_count / words_per_minute
        return duration_minutes * 60.0
    
    def _generate_cache_key(self, request: SpeechRequest) -> str:
        """Generate cache key for speech request."""
        import hashlib
        
        key_data = f"{request.text}_{request.voice_settings.voice_id}_{request.voice_settings.speech_rate}_{request.voice_settings.pitch}_{request.voice_settings.tone.value}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    async def _stop_current_playback(self):
        """Stop current speech playback."""
        if self.current_playback:
            try:
                self.current_playback.terminate()
                await asyncio.wait_for(self.current_playback.wait(), timeout=2.0)
            except asyncio.TimeoutError:
                self.current_playback.kill()
            except Exception as e:
                logger.warning(f"Error stopping playback: {str(e)}")
            finally:
                self.current_playback = None
    
    def _update_synthesis_stats(self, result: Optional[SpeechResult], success: bool):
        """Update synthesis statistics."""
        self.feedback_stats["total_requests"] += 1
        
        if success and result:
            self.feedback_stats["successful_synthesis"] += 1
            
            # Update average synthesis time
            total_successful = self.feedback_stats["successful_synthesis"]
            current_avg_time = self.feedback_stats["average_synthesis_time"]
            new_avg_time = ((current_avg_time * (total_successful - 1)) + result.synthesis_time_ms) / total_successful
            self.feedback_stats["average_synthesis_time"] = new_avg_time
            
            # Update total audio duration
            self.feedback_stats["total_audio_duration"] += result.audio_duration_seconds
        else:
            self.feedback_stats["failed_synthesis"] += 1
    
    async def get_voice_for_language(self, language: VoiceLanguage, gender: Optional[VoiceGender] = None) -> Optional[VoiceSettings]:
        """Get best available voice for specified language and gender."""
        matching_voices = [
            voice for voice in self.available_voices.values()
            if voice.language == language
        ]
        
        if gender:
            gendered_voices = [voice for voice in matching_voices if voice.gender == gender]
            if gendered_voices:
                matching_voices = gendered_voices
        
        return matching_voices[0] if matching_voices else None
    
    async def configure_voice_profile(
        self,
        speaker_id: SpeakerId,
        voice_preferences: Dict[str, Any]
    ) -> Either[VoiceControlError, VoiceSettings]:
        """Configure personalized voice settings for speaker."""
        try:
            # Extract preferences
            preferred_voice = voice_preferences.get("voice_id", "alex")
            speech_rate = voice_preferences.get("speech_rate", 200.0)
            pitch = voice_preferences.get("pitch", 1.0)
            volume = voice_preferences.get("volume", 0.8)
            tone = VoiceTone(voice_preferences.get("tone", "neutral"))
            
            # Validate preferences
            if preferred_voice not in self.available_voices:
                preferred_voice = "alex"  # Fallback to default
            
            base_voice = self.available_voices[preferred_voice]
            
            # Create customized voice settings
            custom_settings = VoiceSettings(
                voice_id=base_voice.voice_id,
                voice_name=base_voice.voice_name,
                language=base_voice.language,
                gender=base_voice.gender,
                speech_rate=max(50.0, min(400.0, speech_rate)),
                pitch=max(0.5, min(2.0, pitch)),
                volume=max(0.0, min(1.0, volume)),
                tone=tone
            )
            
            logger.info(f"Voice profile configured for speaker {speaker_id}: {custom_settings.voice_name}")
            
            return Either.success(custom_settings)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"Voice profile configuration failed: {str(e)}"))
    
    async def provide_command_feedback(
        self,
        command_result: str,
        feedback_type: VoiceTone = VoiceTone.NEUTRAL,
        voice_settings: Optional[VoiceSettings] = None
    ) -> Either[VoiceControlError, SpeechResult]:
        """Provide voice feedback for command execution results."""
        try:
            # Use default voice if none provided
            if not voice_settings:
                voice_settings = self.available_voices["alex"]
            
            # Adjust voice settings for feedback type
            adjusted_settings = voice_settings.adjust_for_tone(feedback_type)
            
            # Create speech request
            request = SpeechRequest(
                text=command_result,
                voice_settings=adjusted_settings,
                output_destination="system",
                interrupt_current=True
            )
            
            return await self.synthesize_speech(request)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"Command feedback failed: {str(e)}"))
    
    async def get_available_voices(self) -> List[Dict[str, Any]]:
        """Get list of available voices with their capabilities."""
        voices = []
        
        for voice_id, voice_settings in self.available_voices.items():
            voices.append({
                "voice_id": voice_id,
                "voice_name": voice_settings.voice_name,
                "language": voice_settings.language.value,
                "gender": voice_settings.gender.value,
                "default_rate": voice_settings.speech_rate,
                "default_pitch": voice_settings.pitch,
                "default_volume": voice_settings.volume
            })
        
        return voices
    
    async def get_feedback_stats(self) -> Dict[str, Any]:
        """Get voice feedback system statistics."""
        stats = self.feedback_stats.copy()
        stats["cache_size"] = len(self.synthesis_cache)
        stats["available_voices"] = len(self.available_voices)
        stats["is_playing"] = self.current_playback is not None
        
        return stats
    
    def clear_cache(self):
        """Clear synthesis result cache."""
        self.synthesis_cache.clear()
        logger.info("Voice synthesis cache cleared")
    
    async def provide_feedback(self, message: str, settings) -> Either[VoiceControlError, Any]:
        """Provide voice feedback with text-to-speech."""
        try:
            # Convert settings to our format
            voice_settings = VoiceSettings(
                voice_id="alex",
                voice_name="Alex",
                language=settings.language,
                gender=VoiceGender.MALE,
                speech_rate=settings.speech_rate * 200.0,
                volume=settings.volume
            )
            
            # Create speech request
            request = SpeechRequest(
                text=message,
                voice_settings=voice_settings,
                output_destination="system",
                save_audio=settings.save_audio
            )
            
            # Synthesize speech
            result = await self.synthesize_speech(request)
            
            if result.is_success():
                return Either.success(result.value)
            else:
                return Either.error(result.error_value)
            
        except Exception as e:
            return Either.error(VoiceControlError(f"Voice feedback failed: {str(e)}"))
    
    async def configure_feedback(self, configuration) -> Either[VoiceControlError, Dict[str, Any]]:
        """Configure voice feedback settings."""
        try:
            result = {
                "feedback_configured": True,
                "voice_settings": configuration.voice_feedback_settings
            }
            return Either.success(result)
        except Exception as e:
            return Either.error(VoiceControlError(f"Feedback configuration failed: {str(e)}"))


# Alias for compatibility
VoiceFeedbackManager = VoiceFeedbackSystem


# Helper functions for voice feedback
def create_success_feedback(message: str) -> str:
    """Create success feedback message."""
    return f"✓ {message}. Command completed successfully."


def create_error_feedback(error: str) -> str:
    """Create error feedback message."""
    return f"⚠ Error: {error}. Please try again."


def create_confirmation_request(action: str) -> str:
    """Create confirmation request message."""
    return f"Are you sure you want to {action}? Say 'yes' to confirm or 'no' to cancel."


def create_progress_feedback(action: str, progress: int, total: int) -> str:
    """Create progress feedback message."""
    percentage = int((progress / total) * 100) if total > 0 else 0
    return f"{action} in progress: {percentage}% complete ({progress} of {total})."