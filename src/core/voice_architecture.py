"""
Voice Control Architecture - TASK_66 Phase 1 Architecture & Design

Voice command recognition, speech processing, and automation control architecture
with enterprise-grade type safety and security boundaries.

Architecture: Voice Engine + Speech Recognition + Intent Processing + Command Dispatcher
Performance: <200ms voice recognition, <100ms command processing, <500ms execution
Security: Voice command validation, speaker authentication, secure audio processing
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Union, Callable
from datetime import datetime, UTC, timedelta
from dataclasses import dataclass, field
from enum import Enum
import uuid
import logging

from .either import Either
from .contracts import require, ensure
from .errors import ValidationError, SecurityError, SystemError

logger = logging.getLogger(__name__)


class VoiceCommandType(Enum):
    """Types of voice commands."""
    AUTOMATION_TRIGGER = "automation_trigger"
    MACRO_EXECUTION = "macro_execution"
    SYSTEM_CONTROL = "system_control"
    APPLICATION_CONTROL = "application_control"
    TEXT_INPUT = "text_input"
    NAVIGATION = "navigation"
    ACCESSIBILITY = "accessibility"
    CUSTOM_WORKFLOW = "custom_workflow"


class SpeechRecognitionEngine(Enum):
    """Supported speech recognition engines."""
    SYSTEM_NATIVE = "system_native"  # macOS native speech recognition
    OPENAI_WHISPER = "openai_whisper"  # OpenAI Whisper API
    GOOGLE_SPEECH = "google_speech"  # Google Speech-to-Text
    AZURE_SPEECH = "azure_speech"  # Azure Speech Services
    LOCAL_WHISPER = "local_whisper"  # Local Whisper model
    AUTO_SELECT = "auto_select"  # Automatically select best engine


class VoiceLanguage(Enum):
    """Supported voice recognition languages."""
    ENGLISH_US = "en-US"
    ENGLISH_UK = "en-GB"
    ENGLISH_AU = "en-AU"
    SPANISH_ES = "es-ES"
    SPANISH_MX = "es-MX"
    FRENCH_FR = "fr-FR"
    GERMAN_DE = "de-DE"
    ITALIAN_IT = "it-IT"
    PORTUGUESE_PT = "pt-PT"
    JAPANESE_JP = "ja-JP"
    KOREAN_KR = "ko-KR"
    CHINESE_CN = "zh-CN"


class CommandPriority(Enum):
    """Voice command execution priority levels."""
    EMERGENCY = "emergency"  # Immediate execution (safety commands)
    HIGH = "high"  # Quick execution (system commands)
    MEDIUM = "medium"  # Normal execution (automation)
    LOW = "low"  # Background execution (non-critical)
    DEFER = "defer"  # Deferred execution (batch operations)


class SpeakerAuthLevel(Enum):
    """Speaker authentication levels."""
    NONE = "none"  # No authentication required
    BASIC = "basic"  # Basic voice pattern matching
    BIOMETRIC = "biometric"  # Advanced biometric authentication
    MULTI_FACTOR = "multi_factor"  # Voice + additional factor
    ENTERPRISE = "enterprise"  # Enterprise-grade authentication


class TrainingType(Enum):
    """Voice recognition training types."""
    USER_VOICE = "user_voice"
    CUSTOM_COMMANDS = "custom_commands"
    ACCENT_ADAPTATION = "accent_adaptation"


VoiceCommandId = str
SpeakerId = str
RecognitionSessionId = str
AudioStreamId = str


@dataclass(frozen=True)
class VoiceProfile:
    """Speaker voice profile for personalization and authentication."""
    speaker_id: SpeakerId
    name: str
    voice_patterns: Dict[str, Any]
    language_preference: VoiceLanguage
    authentication_level: SpeakerAuthLevel
    custom_commands: Dict[str, str] = field(default_factory=dict)
    accessibility_settings: Dict[str, Any] = field(default_factory=dict)
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    last_used: Optional[datetime] = None
    
    @require(lambda self: len(self.speaker_id) > 0)
    @require(lambda self: len(self.name) > 0)
    def __post_init__(self):
        pass
    
    def supports_language(self, language: VoiceLanguage) -> bool:
        """Check if profile supports specified language."""
        return self.language_preference == language
    
    def get_custom_command(self, phrase: str) -> Optional[str]:
        """Get custom command mapping for phrase."""
        return self.custom_commands.get(phrase.lower())
    
    def requires_authentication(self) -> bool:
        """Check if profile requires authentication."""
        return self.authentication_level != SpeakerAuthLevel.NONE


@dataclass(frozen=True)
class AudioInput:
    """Audio input specification for voice processing."""
    audio_data: Optional[bytes]
    audio_file_path: Optional[str]
    sample_rate: int = 16000
    channels: int = 1
    bit_depth: int = 16
    duration_seconds: Optional[float] = None
    noise_level: Optional[float] = None
    
    @require(lambda self: self.sample_rate > 0)
    @require(lambda self: self.channels > 0)
    @require(lambda self: self.bit_depth > 0)
    @require(lambda self: self.audio_data is not None or self.audio_file_path is not None)
    def __post_init__(self):
        pass
    
    def is_file_input(self) -> bool:
        """Check if input is from file."""
        return self.audio_file_path is not None
    
    def is_stream_input(self) -> bool:
        """Check if input is from audio stream."""
        return self.audio_data is not None
    
    def get_audio_info(self) -> Dict[str, Any]:
        """Get audio technical information."""
        return {
            "sample_rate": self.sample_rate,
            "channels": self.channels,
            "bit_depth": self.bit_depth,
            "duration": self.duration_seconds,
            "noise_level": self.noise_level,
            "input_type": "file" if self.is_file_input() else "stream"
        }


@dataclass(frozen=True)
class RecognitionSettings:
    """Speech recognition configuration settings."""
    engine: SpeechRecognitionEngine
    language: VoiceLanguage
    confidence_threshold: float = 0.8
    enable_noise_filtering: bool = True
    enable_echo_cancellation: bool = True
    enable_speaker_identification: bool = False
    enable_continuous_listening: bool = False
    wake_word: Optional[str] = None
    recognition_timeout: timedelta = field(default_factory=lambda: timedelta(seconds=10))
    
    @require(lambda self: 0.0 <= self.confidence_threshold <= 1.0)
    def __post_init__(self):
        pass
    
    def is_wake_word_enabled(self) -> bool:
        """Check if wake word detection is enabled."""
        return self.wake_word is not None and len(self.wake_word) > 0
    
    def get_timeout_seconds(self) -> float:
        """Get recognition timeout in seconds."""
        return self.recognition_timeout.total_seconds()


@dataclass
class VoiceRecognitionResult:
    """Speech recognition result with confidence and alternatives."""
    recognized_text: str
    confidence: float
    language_detected: VoiceLanguage
    recognition_time_ms: float
    alternatives: List[str] = field(default_factory=list)
    speaker_id: Optional[SpeakerId] = None
    audio_info: Optional[Dict[str, Any]] = None
    
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    @require(lambda self: self.recognition_time_ms >= 0.0)
    def __post_init__(self):
        pass
    
    def is_high_confidence(self, threshold: float = 0.8) -> bool:
        """Check if recognition meets confidence threshold."""
        return self.confidence >= threshold
    
    def has_speaker_identification(self) -> bool:
        """Check if speaker was identified."""
        return self.speaker_id is not None
    
    def get_best_alternative(self) -> Optional[str]:
        """Get the best alternative recognition if available."""
        return self.alternatives[0] if self.alternatives else None


@dataclass(frozen=True)
class VoiceCommand:
    """Parsed voice command with intent and parameters."""
    command_id: VoiceCommandId
    command_type: VoiceCommandType
    intent: str
    parameters: Dict[str, Any]
    original_text: str
    confidence: float
    priority: CommandPriority = CommandPriority.MEDIUM
    speaker_id: Optional[SpeakerId] = None
    requires_confirmation: bool = False
    
    @require(lambda self: len(self.command_id) > 0)
    @require(lambda self: len(self.intent) > 0)
    @require(lambda self: 0.0 <= self.confidence <= 1.0)
    def __post_init__(self):
        pass
    
    def is_high_priority(self) -> bool:
        """Check if command is high priority."""
        return self.priority in [CommandPriority.EMERGENCY, CommandPriority.HIGH]
    
    def needs_confirmation(self) -> bool:
        """Check if command requires user confirmation."""
        return self.requires_confirmation or self.priority == CommandPriority.EMERGENCY
    
    def get_parameter(self, key: str, default: Any = None) -> Any:
        """Get command parameter with optional default."""
        return self.parameters.get(key, default)


@dataclass
class VoiceCommandExecution:
    """Voice command execution result with feedback."""
    command_id: VoiceCommandId
    execution_status: str  # "success", "failed", "pending", "cancelled"
    result_data: Optional[Dict[str, Any]] = None
    error_message: Optional[str] = None
    execution_time_ms: float = 0.0
    automation_triggered: Optional[str] = None
    voice_feedback: Optional[str] = None
    
    @require(lambda self: len(self.command_id) > 0)
    @require(lambda self: self.execution_time_ms >= 0.0)
    def __post_init__(self):
        pass
    
    def is_successful(self) -> bool:
        """Check if execution was successful."""
        return self.execution_status == "success"
    
    def has_error(self) -> bool:
        """Check if execution had an error."""
        return self.error_message is not None
    
    def should_provide_feedback(self) -> bool:
        """Check if voice feedback should be provided."""
        return self.voice_feedback is not None


@dataclass(frozen=True)
class VoiceControlSession:
    """Voice control session for managing continuous interaction."""
    session_id: RecognitionSessionId
    speaker_profile: Optional[VoiceProfile]
    recognition_settings: RecognitionSettings
    created_at: datetime
    last_activity: datetime
    active_commands: List[VoiceCommandId] = field(default_factory=list)
    session_context: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.session_id) > 0)
    def __post_init__(self):
        pass
    
    def is_active(self, timeout_minutes: int = 30) -> bool:
        """Check if session is still active."""
        timeout = timedelta(minutes=timeout_minutes)
        return datetime.now(UTC) - self.last_activity < timeout
    
    def has_speaker_profile(self) -> bool:
        """Check if session has speaker profile."""
        return self.speaker_profile is not None
    
    def get_context_value(self, key: str, default: Any = None) -> Any:
        """Get session context value."""
        return self.session_context.get(key, default)


# Voice processing errors
class VoiceControlError(SystemError):
    """Base exception for voice control errors."""
    pass


class SpeechRecognitionError(VoiceControlError):
    """Speech recognition specific errors."""
    
    @classmethod
    def recognition_failed(cls, reason: str) -> 'SpeechRecognitionError':
        return cls(f"Speech recognition failed: {reason}")
    
    @classmethod
    def audio_input_invalid(cls, details: str) -> 'SpeechRecognitionError':
        return cls(f"Invalid audio input: {details}")
    
    @classmethod
    def engine_unavailable(cls, engine: SpeechRecognitionEngine) -> 'SpeechRecognitionError':
        return cls(f"Speech recognition engine unavailable: {engine.value}")
    
    @classmethod
    def confidence_too_low(cls, confidence: float, threshold: float) -> 'SpeechRecognitionError':
        return cls(f"Recognition confidence too low: {confidence} < {threshold}")


class VoiceCommandError(VoiceControlError):
    """Voice command processing errors."""
    
    @classmethod
    def intent_not_recognized(cls, text: str) -> 'VoiceCommandError':
        return cls(f"Could not recognize intent in: {text}")
    
    @classmethod
    def command_execution_failed(cls, command_id: str, reason: str) -> 'VoiceCommandError':
        return cls(f"Command {command_id} execution failed: {reason}")
    
    @classmethod
    def speaker_not_authorized(cls, speaker_id: str, command: str) -> 'VoiceCommandError':
        return cls(f"Speaker {speaker_id} not authorized for command: {command}")
    
    @classmethod
    def unsafe_command_detected(cls, command: str) -> 'VoiceCommandError':
        return cls(f"Unsafe command detected: {command}")


class VoiceAuthenticationError(VoiceControlError):
    """Speaker authentication errors."""
    
    @classmethod
    def speaker_not_recognized(cls) -> 'VoiceAuthenticationError':
        return cls("Speaker voice pattern not recognized")
    
    @classmethod
    def authentication_required(cls, command_type: str) -> 'VoiceAuthenticationError':
        return cls(f"Authentication required for command type: {command_type}")
    
    @classmethod
    def biometric_verification_failed(cls) -> 'VoiceAuthenticationError':
        return cls("Biometric voice verification failed")


# Helper functions for voice architecture
def create_voice_command_id() -> VoiceCommandId:
    """Generate unique voice command ID."""
    return f"voice_cmd_{uuid.uuid4().hex[:12]}"


def create_speaker_id(name: str) -> SpeakerId:
    """Generate speaker ID from name."""
    name_clean = "".join(c for c in name.lower() if c.isalnum())
    return f"speaker_{name_clean}_{uuid.uuid4().hex[:8]}"


def create_session_id() -> RecognitionSessionId:
    """Generate unique voice session ID."""
    return f"voice_session_{uuid.uuid4().hex[:16]}"


def validate_audio_input_security(audio_input: AudioInput) -> Either[VoiceControlError, None]:
    """Validate audio input for security compliance."""
    try:
        # Validate file path if provided
        if audio_input.audio_file_path:
            from pathlib import Path
            
            audio_path = Path(audio_input.audio_file_path)
            
            # Check for path traversal
            if '..' in str(audio_path):
                return Either.error(VoiceControlError("Path traversal detected in audio file"))
            
            # Validate file extension
            allowed_extensions = {'.wav', '.mp3', '.m4a', '.aiff', '.flac'}
            if audio_path.suffix.lower() not in allowed_extensions:
                return Either.error(VoiceControlError("Unsupported audio file format"))
            
            # Check file size (max 50MB)
            if audio_path.exists() and audio_path.stat().st_size > 50 * 1024 * 1024:
                return Either.error(VoiceControlError("Audio file too large"))
        
        # Validate audio data if provided
        if audio_input.audio_data:
            # Check audio data size (max 50MB)
            if len(audio_input.audio_data) > 50 * 1024 * 1024:
                return Either.error(VoiceControlError("Audio data too large"))
        
        # Validate technical parameters
        if not (8000 <= audio_input.sample_rate <= 48000):
            return Either.error(VoiceControlError("Invalid sample rate"))
        
        if not (1 <= audio_input.channels <= 2):
            return Either.error(VoiceControlError("Invalid channel count"))
        
        if not (8 <= audio_input.bit_depth <= 32):
            return Either.error(VoiceControlError("Invalid bit depth"))
        
        return Either.success(None)
        
    except Exception as e:
        return Either.error(VoiceControlError(f"Audio validation failed: {str(e)}"))


def validate_voice_command_security(command: VoiceCommand) -> Either[VoiceControlError, None]:
    """Validate voice command for security and safety."""
    try:
        # Check for dangerous command patterns
        dangerous_patterns = [
            r'(?i)(delete|remove|erase).*file',
            r'(?i)(format|wipe).*disk',
            r'(?i)(shutdown|restart).*system',
            r'(?i)(install|download).*software',
            r'(?i)(execute|run).*script',
            r'(?i)(access|connect).*network',
            r'(?i)(password|credential|key)',
        ]
        
        import re
        text_to_check = f"{command.intent} {command.original_text}".lower()
        
        for pattern in dangerous_patterns:
            if re.search(pattern, text_to_check):
                return Either.error(VoiceCommandError.unsafe_command_detected(command.intent))
        
        # Validate parameter safety
        for key, value in command.parameters.items():
            if isinstance(value, str):
                # Check for injection patterns
                injection_patterns = [
                    r'[;&|`$()]',  # Shell injection characters
                    r'<script',    # Script injection
                    r'javascript:', # JavaScript injection
                ]
                
                for pattern in injection_patterns:
                    if re.search(pattern, value.lower()):
                        return Either.error(VoiceCommandError.unsafe_command_detected(f"Parameter {key}"))
        
        return Either.success(None)
        
    except Exception as e:
        return Either.error(VoiceControlError(f"Command validation failed: {str(e)}"))


def estimate_recognition_cost(audio_input: AudioInput, engine: SpeechRecognitionEngine) -> float:
    """Estimate cost for speech recognition processing."""
    # Base cost estimation (placeholder values)
    cost_per_minute = {
        SpeechRecognitionEngine.SYSTEM_NATIVE: 0.0,  # Free
        SpeechRecognitionEngine.OPENAI_WHISPER: 0.006,  # $0.006 per minute
        SpeechRecognitionEngine.GOOGLE_SPEECH: 0.004,  # $0.004 per minute
        SpeechRecognitionEngine.AZURE_SPEECH: 0.003,  # $0.003 per minute
        SpeechRecognitionEngine.LOCAL_WHISPER: 0.0,  # Free but resource intensive
    }
    
    # Estimate duration if not provided
    duration = audio_input.duration_seconds
    if duration is None and audio_input.audio_data:
        # Rough estimation based on data size
        bytes_per_second = (audio_input.sample_rate * audio_input.channels * audio_input.bit_depth) // 8
        duration = len(audio_input.audio_data) / bytes_per_second
    
    if duration is None:
        duration = 30.0  # Default estimation
    
    duration_minutes = duration / 60.0
    base_cost = cost_per_minute.get(engine, 0.005)
    
    return duration_minutes * base_cost