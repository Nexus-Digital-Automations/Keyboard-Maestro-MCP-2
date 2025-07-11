"""Comprehensive tests for voice architecture module.

Covers voice command processing, speech recognition, and audio handling
with property-based testing and error path validation.
"""

from datetime import UTC, datetime, timedelta
from unittest.mock import patch

import pytest
from hypothesis import assume, given
from hypothesis import strategies as st
from src.core.errors import ContractViolationError
from src.core.voice_architecture import (
    AudioInput,
    CommandPriority,
    RecognitionSettings,
    SpeakerAuthLevel,
    SpeechRecognitionEngine,
    SpeechRecognitionError,
    TrainingType,
    VoiceAuthenticationError,
    VoiceCommand,
    VoiceCommandError,
    VoiceCommandExecution,
    VoiceCommandType,
    VoiceControlSession,
    VoiceLanguage,
    VoiceProfile,
    VoiceRecognitionResult,
    create_session_id,
    create_speaker_id,
    create_voice_command_id,
    estimate_recognition_cost,
    validate_audio_input_security,
    validate_voice_command_security,
)


class TestVoiceEnumerations:
    """Test voice control enumeration classes."""

    def test_voice_command_type_values(self):
        """Test all voice command type values."""
        assert VoiceCommandType.AUTOMATION_TRIGGER.value == "automation_trigger"
        assert VoiceCommandType.MACRO_EXECUTION.value == "macro_execution"
        assert VoiceCommandType.SYSTEM_CONTROL.value == "system_control"
        assert VoiceCommandType.APPLICATION_CONTROL.value == "application_control"
        assert VoiceCommandType.TEXT_INPUT.value == "text_input"
        assert VoiceCommandType.NAVIGATION.value == "navigation"
        assert VoiceCommandType.ACCESSIBILITY.value == "accessibility"
        assert VoiceCommandType.CUSTOM_WORKFLOW.value == "custom_workflow"

    def test_speech_recognition_engine_values(self):
        """Test speech recognition engine values."""
        assert SpeechRecognitionEngine.SYSTEM_NATIVE.value == "system_native"
        assert SpeechRecognitionEngine.OPENAI_WHISPER.value == "openai_whisper"
        assert SpeechRecognitionEngine.GOOGLE_SPEECH.value == "google_speech"
        assert SpeechRecognitionEngine.AZURE_SPEECH.value == "azure_speech"
        assert SpeechRecognitionEngine.LOCAL_WHISPER.value == "local_whisper"
        assert SpeechRecognitionEngine.AUTO_SELECT.value == "auto_select"

    def test_voice_language_values(self):
        """Test voice language enumeration."""
        assert VoiceLanguage.ENGLISH_US.value == "en-US"
        assert VoiceLanguage.ENGLISH_UK.value == "en-GB"
        assert VoiceLanguage.SPANISH_ES.value == "es-ES"
        assert VoiceLanguage.FRENCH_FR.value == "fr-FR"
        assert VoiceLanguage.GERMAN_DE.value == "de-DE"
        assert VoiceLanguage.JAPANESE_JP.value == "ja-JP"
        assert VoiceLanguage.CHINESE_CN.value == "zh-CN"

    def test_command_priority_values(self):
        """Test command priority levels."""
        assert CommandPriority.EMERGENCY.value == "emergency"
        assert CommandPriority.HIGH.value == "high"
        assert CommandPriority.MEDIUM.value == "medium"
        assert CommandPriority.LOW.value == "low"
        assert CommandPriority.DEFER.value == "defer"

    def test_speaker_auth_level_values(self):
        """Test speaker authentication levels."""
        assert SpeakerAuthLevel.NONE.value == "none"
        assert SpeakerAuthLevel.BASIC.value == "basic"
        assert SpeakerAuthLevel.MULTI_FACTOR.value == "multi_factor"
        assert SpeakerAuthLevel.ENTERPRISE.value == "enterprise"

    def test_training_type_values(self):
        """Test training type enumeration."""
        assert TrainingType.USER_VOICE.value == "user_voice"
        assert TrainingType.CUSTOM_COMMANDS.value == "custom_commands"
        assert TrainingType.ACCENT_ADAPTATION.value == "accent_adaptation"


class TestVoiceProfile:
    """Test VoiceProfile dataclass functionality."""

    def test_voice_profile_creation(self):
        """Test valid voice profile creation."""
        now = datetime.now(UTC)
        profile = VoiceProfile(
            profile_id="test_profile_123",
            user_name="Test User",
            acoustic_characteristics={"frequency_range": "80-8000Hz"},
            personalization_level=0.8,
            supported_languages=[VoiceLanguage.ENGLISH_US, VoiceLanguage.SPANISH_ES],
            created_date=now,
            last_updated=now,
            authentication_level=SpeakerAuthLevel.BASIC,
            custom_commands={"lights on": "turn_on_lights"},
        )

        assert profile.profile_id == "test_profile_123"
        assert profile.user_name == "Test User"
        assert profile.personalization_level == 0.8
        assert len(profile.supported_languages) == 2
        assert profile.authentication_level == SpeakerAuthLevel.BASIC

    def test_voice_profile_supports_language(self):
        """Test language support checking."""
        profile = VoiceProfile(
            profile_id="test_profile",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=0.5,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=datetime.now(UTC),
            last_updated=datetime.now(UTC),
        )

        assert profile.supports_language(VoiceLanguage.ENGLISH_US)
        assert not profile.supports_language(VoiceLanguage.FRENCH_FR)

    def test_voice_profile_custom_commands(self):
        """Test custom command functionality."""
        profile = VoiceProfile(
            profile_id="test_profile",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=0.5,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=datetime.now(UTC),
            last_updated=datetime.now(UTC),
            custom_commands={
                "hey computer": "wake_up",
                "lights on": "turn_on_lights",
            },  # Store keys in lowercase
        )

        assert profile.get_custom_command("hey computer") == "wake_up"
        assert (
            profile.get_custom_command("LIGHTS ON") == "turn_on_lights"
        )  # Case insensitive query
        assert profile.get_custom_command("unknown") is None

    def test_voice_profile_authentication_requirements(self):
        """Test authentication requirement checking."""
        profile_no_auth = VoiceProfile(
            profile_id="test_profile",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=0.5,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=datetime.now(UTC),
            last_updated=datetime.now(UTC),
            authentication_level=SpeakerAuthLevel.NONE,
        )

        profile_with_auth = VoiceProfile(
            profile_id="test_profile",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=0.5,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=datetime.now(UTC),
            last_updated=datetime.now(UTC),
            authentication_level=SpeakerAuthLevel.BASIC,
        )

        assert not profile_no_auth.requires_authentication()
        assert profile_with_auth.requires_authentication()

    def test_voice_profile_validation_errors(self):
        """Test voice profile validation failures."""
        now = datetime.now(UTC)

        # Empty profile ID should fail
        with pytest.raises(ContractViolationError):
            VoiceProfile(
                profile_id="",
                user_name="Test User",
                acoustic_characteristics={},
                personalization_level=0.5,
                supported_languages=[VoiceLanguage.ENGLISH_US],
                created_date=now,
                last_updated=now,
            )

        # Empty user name should fail
        with pytest.raises(ContractViolationError):
            VoiceProfile(
                profile_id="test_profile",
                user_name="",
                acoustic_characteristics={},
                personalization_level=0.5,
                supported_languages=[VoiceLanguage.ENGLISH_US],
                created_date=now,
                last_updated=now,
            )

        # Invalid personalization level should fail
        with pytest.raises(ContractViolationError):
            VoiceProfile(
                profile_id="test_profile",
                user_name="Test User",
                acoustic_characteristics={},
                personalization_level=1.5,  # Invalid: > 1.0
                supported_languages=[VoiceLanguage.ENGLISH_US],
                created_date=now,
                last_updated=now,
            )


class TestAudioInput:
    """Test AudioInput dataclass functionality."""

    def test_audio_input_file_creation(self):
        """Test audio input from file."""
        audio_input = AudioInput(
            audio_data=None,
            audio_file_path="/path/to/audio.wav",
            sample_rate=44100,
            channels=2,
            bit_depth=16,
            duration_seconds=10.5,
        )

        assert audio_input.is_file_input()
        assert not audio_input.is_stream_input()
        assert audio_input.sample_rate == 44100
        assert audio_input.channels == 2

    def test_audio_input_stream_creation(self):
        """Test audio input from stream data."""
        audio_data = b"fake_audio_data" * 1000
        audio_input = AudioInput(
            audio_data=audio_data,
            audio_file_path=None,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
        )

        assert audio_input.is_stream_input()
        assert not audio_input.is_file_input()
        assert len(audio_input.audio_data) == len(audio_data)

    def test_audio_input_info(self):
        """Test audio information retrieval."""
        audio_input = AudioInput(
            audio_data=b"test_data",
            audio_file_path=None,
            sample_rate=22050,
            channels=1,
            bit_depth=24,
            duration_seconds=5.0,
            noise_level=0.1,
        )

        info = audio_input.get_audio_info()
        assert info["sample_rate"] == 22050
        assert info["channels"] == 1
        assert info["bit_depth"] == 24
        assert info["duration"] == 5.0
        assert info["noise_level"] == 0.1
        assert info["input_type"] == "stream"

    def test_audio_input_validation_errors(self):
        """Test audio input validation failures."""
        # No audio data or file path
        with pytest.raises(ContractViolationError):
            AudioInput(
                audio_data=None,
                audio_file_path=None,
            )

        # Invalid sample rate
        with pytest.raises(ContractViolationError):
            AudioInput(
                audio_data=b"test",
                audio_file_path=None,
                sample_rate=0,
            )

        # Invalid channels
        with pytest.raises(ContractViolationError):
            AudioInput(
                audio_data=b"test",
                audio_file_path=None,
                channels=0,
            )

        # Invalid bit depth
        with pytest.raises(ContractViolationError):
            AudioInput(
                audio_data=b"test",
                audio_file_path=None,
                bit_depth=0,
            )


class TestRecognitionSettings:
    """Test RecognitionSettings dataclass functionality."""

    def test_recognition_settings_creation(self):
        """Test recognition settings creation."""
        settings = RecognitionSettings(
            engine=SpeechRecognitionEngine.OPENAI_WHISPER,
            language=VoiceLanguage.ENGLISH_US,
            confidence_threshold=0.9,
            enable_noise_filtering=True,
            wake_word="hey assistant",
        )

        assert settings.engine == SpeechRecognitionEngine.OPENAI_WHISPER
        assert settings.language == VoiceLanguage.ENGLISH_US
        assert settings.confidence_threshold == 0.9
        assert settings.is_wake_word_enabled()

    def test_recognition_settings_wake_word(self):
        """Test wake word functionality."""
        settings_with_wake = RecognitionSettings(
            engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
            language=VoiceLanguage.ENGLISH_US,
            wake_word="computer",
        )

        settings_without_wake = RecognitionSettings(
            engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
            language=VoiceLanguage.ENGLISH_US,
            wake_word=None,
        )

        settings_empty_wake = RecognitionSettings(
            engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
            language=VoiceLanguage.ENGLISH_US,
            wake_word="",
        )

        assert settings_with_wake.is_wake_word_enabled()
        assert not settings_without_wake.is_wake_word_enabled()
        assert not settings_empty_wake.is_wake_word_enabled()

    def test_recognition_settings_timeout(self):
        """Test timeout functionality."""
        custom_timeout = timedelta(seconds=15)
        settings = RecognitionSettings(
            engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
            language=VoiceLanguage.ENGLISH_US,
            recognition_timeout=custom_timeout,
        )

        assert settings.get_timeout_seconds() == 15.0

    def test_recognition_settings_validation(self):
        """Test recognition settings validation."""
        # Invalid confidence threshold
        with pytest.raises(ContractViolationError):
            RecognitionSettings(
                engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
                language=VoiceLanguage.ENGLISH_US,
                confidence_threshold=1.5,  # Invalid: > 1.0
            )


class TestVoiceRecognitionResult:
    """Test VoiceRecognitionResult dataclass functionality."""

    def test_recognition_result_creation(self):
        """Test voice recognition result creation."""
        result = VoiceRecognitionResult(
            recognized_text="turn on the lights",
            confidence=0.95,
            language_detected=VoiceLanguage.ENGLISH_US,
            recognition_time_ms=150.0,
            alternatives=["turn on lights", "turn on the light"],
            speaker_id="speaker_123",
        )

        assert result.recognized_text == "turn on the lights"
        assert result.confidence == 0.95
        assert result.is_high_confidence()
        assert result.has_speaker_identification()

    def test_recognition_result_confidence_check(self):
        """Test confidence threshold checking."""
        high_confidence = VoiceRecognitionResult(
            recognized_text="test",
            confidence=0.9,
            language_detected=VoiceLanguage.ENGLISH_US,
            recognition_time_ms=100.0,
        )

        low_confidence = VoiceRecognitionResult(
            recognized_text="test",
            confidence=0.6,
            language_detected=VoiceLanguage.ENGLISH_US,
            recognition_time_ms=100.0,
        )

        assert high_confidence.is_high_confidence(threshold=0.8)
        assert not low_confidence.is_high_confidence(threshold=0.8)

    def test_recognition_result_alternatives(self):
        """Test alternative recognition results."""
        result_with_alternatives = VoiceRecognitionResult(
            recognized_text="primary result",
            confidence=0.8,
            language_detected=VoiceLanguage.ENGLISH_US,
            recognition_time_ms=100.0,
            alternatives=["alternative 1", "alternative 2"],
        )

        result_no_alternatives = VoiceRecognitionResult(
            recognized_text="only result",
            confidence=0.8,
            language_detected=VoiceLanguage.ENGLISH_US,
            recognition_time_ms=100.0,
        )

        assert result_with_alternatives.get_best_alternative() == "alternative 1"
        assert result_no_alternatives.get_best_alternative() is None

    def test_recognition_result_validation(self):
        """Test recognition result validation."""
        # Invalid confidence
        with pytest.raises(ContractViolationError):
            VoiceRecognitionResult(
                recognized_text="test",
                confidence=1.5,  # Invalid: > 1.0
                language_detected=VoiceLanguage.ENGLISH_US,
                recognition_time_ms=100.0,
            )

        # Invalid recognition time
        with pytest.raises(ContractViolationError):
            VoiceRecognitionResult(
                recognized_text="test",
                confidence=0.8,
                language_detected=VoiceLanguage.ENGLISH_US,
                recognition_time_ms=-10.0,  # Invalid: negative
            )


class TestVoiceCommand:
    """Test VoiceCommand dataclass functionality."""

    def test_voice_command_creation(self):
        """Test voice command creation."""
        command = VoiceCommand(
            command_id="cmd_123",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="turn_on_lights",
            parameters={"room": "living room", "brightness": 80},
            original_text="turn on the living room lights to 80 percent",
            confidence=0.92,
            priority=CommandPriority.HIGH,
            speaker_id="speaker_456",
            requires_confirmation=True,
        )

        assert command.command_id == "cmd_123"
        assert command.command_type == VoiceCommandType.AUTOMATION_TRIGGER
        assert command.intent == "turn_on_lights"
        assert command.is_high_priority()
        assert command.needs_confirmation()

    def test_voice_command_priority_checking(self):
        """Test command priority classification."""
        emergency_cmd = VoiceCommand(
            command_id="emergency_cmd",
            command_type=VoiceCommandType.SYSTEM_CONTROL,
            intent="emergency_stop",
            parameters={},
            original_text="emergency stop",
            confidence=0.95,
            priority=CommandPriority.EMERGENCY,
        )

        high_priority_cmd = VoiceCommand(
            command_id="high_cmd",
            command_type=VoiceCommandType.SYSTEM_CONTROL,
            intent="system_shutdown",
            parameters={},
            original_text="shut down system",
            confidence=0.9,
            priority=CommandPriority.HIGH,
        )

        medium_priority_cmd = VoiceCommand(
            command_id="medium_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="adjust_temperature",
            parameters={},
            original_text="set temperature to 72",
            confidence=0.85,
            priority=CommandPriority.MEDIUM,
        )

        assert emergency_cmd.is_high_priority()
        assert high_priority_cmd.is_high_priority()
        assert not medium_priority_cmd.is_high_priority()

    def test_voice_command_confirmation_logic(self):
        """Test command confirmation requirements."""
        emergency_cmd = VoiceCommand(
            command_id="emergency_cmd",
            command_type=VoiceCommandType.SYSTEM_CONTROL,
            intent="emergency_stop",
            parameters={},
            original_text="emergency stop",
            confidence=0.95,
            priority=CommandPriority.EMERGENCY,
            requires_confirmation=False,  # Even without explicit flag, emergency needs confirmation
        )

        explicit_confirmation_cmd = VoiceCommand(
            command_id="confirm_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="delete_file",
            parameters={},
            original_text="delete my file",
            confidence=0.9,
            priority=CommandPriority.MEDIUM,
            requires_confirmation=True,
        )

        normal_cmd = VoiceCommand(
            command_id="normal_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="turn_on_lights",
            parameters={},
            original_text="turn on lights",
            confidence=0.9,
            priority=CommandPriority.MEDIUM,
            requires_confirmation=False,
        )

        assert emergency_cmd.needs_confirmation()  # Emergency always needs confirmation
        assert explicit_confirmation_cmd.needs_confirmation()
        assert not normal_cmd.needs_confirmation()

    def test_voice_command_parameters(self):
        """Test command parameter handling."""
        command = VoiceCommand(
            command_id="param_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="set_temperature",
            parameters={"temperature": 72, "unit": "fahrenheit", "room": "bedroom"},
            original_text="set bedroom temperature to 72 degrees",
            confidence=0.9,
        )

        assert command.get_parameter("temperature") == 72
        assert command.get_parameter("unit") == "fahrenheit"
        assert command.get_parameter("nonexistent") is None
        assert command.get_parameter("nonexistent", "default") == "default"

    def test_voice_command_validation(self):
        """Test voice command validation."""
        # Empty command ID
        with pytest.raises(ContractViolationError):
            VoiceCommand(
                command_id="",
                command_type=VoiceCommandType.AUTOMATION_TRIGGER,
                intent="test",
                parameters={},
                original_text="test",
                confidence=0.8,
            )

        # Empty intent
        with pytest.raises(ContractViolationError):
            VoiceCommand(
                command_id="cmd_123",
                command_type=VoiceCommandType.AUTOMATION_TRIGGER,
                intent="",
                parameters={},
                original_text="test",
                confidence=0.8,
            )

        # Invalid confidence
        with pytest.raises(ContractViolationError):
            VoiceCommand(
                command_id="cmd_123",
                command_type=VoiceCommandType.AUTOMATION_TRIGGER,
                intent="test",
                parameters={},
                original_text="test",
                confidence=1.5,
            )


class TestVoiceCommandExecution:
    """Test VoiceCommandExecution dataclass functionality."""

    def test_command_execution_success(self):
        """Test successful command execution."""
        execution = VoiceCommandExecution(
            command_id="cmd_123",
            execution_status="success",
            result_data={
                "action": "completed",
                "affected_devices": ["light_1", "light_2"],
            },
            execution_time_ms=250.0,
            automation_triggered="home_automation_rule_5",
            voice_feedback="Lights have been turned on",
        )

        assert execution.is_successful()
        assert not execution.has_error()
        assert execution.should_provide_feedback()

    def test_command_execution_failure(self):
        """Test failed command execution."""
        execution = VoiceCommandExecution(
            command_id="cmd_456",
            execution_status="failed",
            error_message="Device not responding",
            execution_time_ms=1000.0,
        )

        assert not execution.is_successful()
        assert execution.has_error()
        assert not execution.should_provide_feedback()

    def test_command_execution_validation(self):
        """Test command execution validation."""
        # Empty command ID
        with pytest.raises(ContractViolationError):
            VoiceCommandExecution(
                command_id="",
                execution_status="success",
            )

        # Negative execution time
        with pytest.raises(ContractViolationError):
            VoiceCommandExecution(
                command_id="cmd_123",
                execution_status="success",
                execution_time_ms=-10.0,
            )


class TestVoiceControlSession:
    """Test VoiceControlSession dataclass functionality."""

    def test_voice_session_creation(self):
        """Test voice control session creation."""
        now = datetime.now(UTC)
        profile = VoiceProfile(
            profile_id="profile_123",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=0.8,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=now,
            last_updated=now,
        )

        settings = RecognitionSettings(
            engine=SpeechRecognitionEngine.OPENAI_WHISPER,
            language=VoiceLanguage.ENGLISH_US,
        )

        session = VoiceControlSession(
            session_id="session_456",
            speaker_profile=profile,
            recognition_settings=settings,
            created_at=now,
            last_activity=now,
            active_commands=["cmd_1", "cmd_2"],
            session_context={"device_state": "active", "room": "office"},
        )

        assert session.session_id == "session_456"
        assert session.has_speaker_profile()
        assert session.is_active()
        assert len(session.active_commands) == 2

    def test_voice_session_activity_tracking(self):
        """Test session activity and timeout logic."""
        now = datetime.now(UTC)

        # Recent activity - should be active
        recent_session = VoiceControlSession(
            session_id="recent_session",
            speaker_profile=None,
            recognition_settings=RecognitionSettings(
                engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
                language=VoiceLanguage.ENGLISH_US,
            ),
            created_at=now,
            last_activity=now - timedelta(minutes=5),  # 5 minutes ago
        )

        # Old activity - should be inactive
        old_session = VoiceControlSession(
            session_id="old_session",
            speaker_profile=None,
            recognition_settings=RecognitionSettings(
                engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
                language=VoiceLanguage.ENGLISH_US,
            ),
            created_at=now,
            last_activity=now - timedelta(minutes=45),  # 45 minutes ago
        )

        assert recent_session.is_active(timeout_minutes=30)
        assert not old_session.is_active(timeout_minutes=30)

    def test_voice_session_context(self):
        """Test session context management."""
        session = VoiceControlSession(
            session_id="context_session",
            speaker_profile=None,
            recognition_settings=RecognitionSettings(
                engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
                language=VoiceLanguage.ENGLISH_US,
            ),
            created_at=datetime.now(UTC),
            last_activity=datetime.now(UTC),
            session_context={
                "user_preference": "quiet_mode",
                "last_command": "lights_off",
            },
        )

        assert session.get_context_value("user_preference") == "quiet_mode"
        assert session.get_context_value("last_command") == "lights_off"
        assert session.get_context_value("nonexistent") is None
        assert session.get_context_value("nonexistent", "default") == "default"

    def test_voice_session_validation(self):
        """Test voice session validation."""
        # Empty session ID
        with pytest.raises(ContractViolationError):
            VoiceControlSession(
                session_id="",
                speaker_profile=None,
                recognition_settings=RecognitionSettings(
                    engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
                    language=VoiceLanguage.ENGLISH_US,
                ),
                created_at=datetime.now(UTC),
                last_activity=datetime.now(UTC),
            )


class TestVoiceErrors:
    """Test voice control error classes."""

    def test_speech_recognition_errors(self):
        """Test speech recognition error creation."""
        recognition_failed = SpeechRecognitionError.recognition_failed(
            "microphone not found"
        )
        assert "Speech recognition failed: microphone not found" in str(
            recognition_failed
        )

        audio_invalid = SpeechRecognitionError.audio_input_invalid("unsupported format")
        assert "Invalid audio input: unsupported format" in str(audio_invalid)

        engine_unavailable = SpeechRecognitionError.engine_unavailable(
            SpeechRecognitionEngine.OPENAI_WHISPER
        )
        assert "Speech recognition engine unavailable: openai_whisper" in str(
            engine_unavailable
        )

        confidence_low = SpeechRecognitionError.confidence_too_low(0.6, 0.8)
        assert "Recognition confidence too low: 0.6 < 0.8" in str(confidence_low)

    def test_voice_command_errors(self):
        """Test voice command error creation."""
        intent_error = VoiceCommandError.intent_not_recognized("mumbled speech")
        assert "Could not recognize intent in: mumbled speech" in str(intent_error)

        execution_error = VoiceCommandError.command_execution_failed(
            "cmd_123", "device offline"
        )
        assert "Command cmd_123 execution failed: device offline" in str(
            execution_error
        )

        auth_error = VoiceCommandError.speaker_not_authorized(
            "speaker_456", "system_shutdown"
        )
        assert "Speaker speaker_456 not authorized for command: system_shutdown" in str(
            auth_error
        )

        unsafe_error = VoiceCommandError.unsafe_command_detected("delete all files")
        assert "Unsafe command detected: delete all files" in str(unsafe_error)

    def test_voice_authentication_errors(self):
        """Test voice authentication error creation."""
        speaker_error = VoiceAuthenticationError.speaker_not_recognized()
        assert "Speaker voice pattern not recognized" in str(speaker_error)

        auth_required_error = VoiceAuthenticationError.authentication_required(
            "system_control"
        )
        assert "Authentication required for command type: system_control" in str(
            auth_required_error
        )


class TestVoiceHelperFunctions:
    """Test voice architecture helper functions."""

    def test_voice_command_id_generation(self):
        """Test voice command ID generation."""
        cmd_id1 = create_voice_command_id()
        cmd_id2 = create_voice_command_id()

        assert cmd_id1.startswith("voice_cmd_")
        assert cmd_id2.startswith("voice_cmd_")
        assert cmd_id1 != cmd_id2  # Should be unique
        assert len(cmd_id1) == len("voice_cmd_") + 12  # voice_cmd_ + 12 hex chars

    def test_speaker_id_generation(self):
        """Test speaker ID generation."""
        speaker_id1 = create_speaker_id("John Doe")
        speaker_id2 = create_speaker_id("Jane Smith")
        speaker_id3 = create_speaker_id("John Doe")  # Same name

        assert speaker_id1.startswith("speaker_johndoe_")
        assert speaker_id2.startswith("speaker_janesmith_")
        assert speaker_id1 != speaker_id3  # Same name but different IDs
        assert all(
            c.isalnum() or c == "_" for c in speaker_id1
        )  # Only alphanumeric and underscore

    def test_session_id_generation(self):
        """Test session ID generation."""
        session_id1 = create_session_id()
        session_id2 = create_session_id()

        assert session_id1.startswith("voice_session_")
        assert session_id2.startswith("voice_session_")
        assert session_id1 != session_id2  # Should be unique
        assert (
            len(session_id1) == len("voice_session_") + 16
        )  # voice_session_ + 16 hex chars


class TestAudioInputSecurity:
    """Test audio input security validation."""

    def test_valid_audio_file_input(self):
        """Test valid audio file input passes security validation."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            # Mock file size (10MB)
            mock_stat.return_value.st_size = 10 * 1024 * 1024

            audio_input = AudioInput(
                audio_data=None,
                audio_file_path="/safe/path/audio.wav",
                sample_rate=44100,
                channels=2,
                bit_depth=16,
            )

            result = validate_audio_input_security(audio_input)
            assert result.is_right()

    def test_path_traversal_detection(self):
        """Test path traversal attack detection."""
        audio_input = AudioInput(
            audio_data=None,
            audio_file_path="/safe/path/../../../etc/passwd",
            sample_rate=44100,
            channels=2,
            bit_depth=16,
        )

        result = validate_audio_input_security(audio_input)
        assert result.is_left()
        assert "Path traversal detected" in str(result.get_left())

    def test_unsupported_file_format(self):
        """Test unsupported file format rejection."""
        audio_input = AudioInput(
            audio_data=None,
            audio_file_path="/safe/path/audio.exe",  # Unsupported format
            sample_rate=44100,
            channels=2,
            bit_depth=16,
        )

        result = validate_audio_input_security(audio_input)
        assert result.is_left()
        assert "Unsupported audio file format" in str(result.get_left())

    def test_file_size_limit(self):
        """Test file size limit enforcement."""
        with (
            patch("pathlib.Path.exists", return_value=True),
            patch("pathlib.Path.stat") as mock_stat,
        ):
            # Mock large file (100MB)
            mock_stat.return_value.st_size = 100 * 1024 * 1024

            audio_input = AudioInput(
                audio_data=None,
                audio_file_path="/safe/path/huge_audio.wav",
                sample_rate=44100,
                channels=2,
                bit_depth=16,
            )

            result = validate_audio_input_security(audio_input)
            assert result.is_left()
            assert "Audio file too large" in str(result.get_left())

    def test_audio_data_size_limit(self):
        """Test audio data size limit."""
        # Create large audio data (60MB)
        large_audio_data = b"x" * (60 * 1024 * 1024)

        audio_input = AudioInput(
            audio_data=large_audio_data,
            audio_file_path=None,
            sample_rate=44100,
            channels=2,
            bit_depth=16,
        )

        result = validate_audio_input_security(audio_input)
        assert result.is_left()
        assert "Audio data too large" in str(result.get_left())

    def test_invalid_audio_parameters(self):
        """Test invalid audio parameter detection."""
        # Invalid sample rate
        audio_input_bad_rate = AudioInput(
            audio_data=b"test_data",
            audio_file_path=None,
            sample_rate=100000,  # Too high
            channels=1,
            bit_depth=16,
        )

        result = validate_audio_input_security(audio_input_bad_rate)
        assert result.is_left()
        assert "Invalid sample rate" in str(result.get_left())

        # Invalid channels
        audio_input_bad_channels = AudioInput(
            audio_data=b"test_data",
            audio_file_path=None,
            sample_rate=44100,
            channels=5,  # Too many
            bit_depth=16,
        )

        result = validate_audio_input_security(audio_input_bad_channels)
        assert result.is_left()
        assert "Invalid channel count" in str(result.get_left())

        # Invalid bit depth
        audio_input_bad_depth = AudioInput(
            audio_data=b"test_data",
            audio_file_path=None,
            sample_rate=44100,
            channels=1,
            bit_depth=64,  # Too high
        )

        result = validate_audio_input_security(audio_input_bad_depth)
        assert result.is_left()
        assert "Invalid bit depth" in str(result.get_left())


class TestVoiceCommandSecurity:
    """Test voice command security validation."""

    def test_safe_voice_command(self):
        """Test safe voice command passes validation."""
        safe_command = VoiceCommand(
            command_id="safe_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="turn_on_lights",
            parameters={"room": "living room", "brightness": 80},
            original_text="turn on the living room lights",
            confidence=0.9,
        )

        result = validate_voice_command_security(safe_command)
        assert result.is_right()

    def test_dangerous_command_patterns(self):
        """Test detection of dangerous command patterns."""
        dangerous_commands = [
            ("delete_file", "delete important file"),
            ("format_disk", "format system disk"),
            ("shutdown_system", "shutdown the system"),
            ("install_software", "install new software"),
            ("execute_script", "execute the script"),
            ("access_network", "access network settings"),
            ("password_manager", "open password manager"),
        ]

        for intent, text in dangerous_commands:
            dangerous_command = VoiceCommand(
                command_id="danger_cmd",
                command_type=VoiceCommandType.SYSTEM_CONTROL,
                intent=intent,
                parameters={},
                original_text=text,
                confidence=0.9,
            )

            result = validate_voice_command_security(dangerous_command)
            assert result.is_left()
            assert "Unsafe command detected" in str(result.get_left())

    def test_parameter_injection_detection(self):
        """Test detection of injection attacks in parameters."""
        injection_command = VoiceCommand(
            command_id="injection_cmd",
            command_type=VoiceCommandType.AUTOMATION_TRIGGER,
            intent="set_text",
            parameters={
                "text": "normal text; rm -rf /",  # Shell injection
                "url": "javascript:alert('xss')",  # JavaScript injection
                "script": "<script>malicious()</script>",  # Script injection
            },
            original_text="set text with injection",
            confidence=0.9,
        )

        result = validate_voice_command_security(injection_command)
        assert result.is_left()
        assert "Unsafe command detected" in str(result.get_left())


class TestCostEstimation:
    """Test speech recognition cost estimation."""

    def test_cost_estimation_system_native(self):
        """Test cost estimation for system native engine (free)."""
        audio_input = AudioInput(
            audio_data=b"test_audio_data" * 1000,
            audio_file_path=None,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_seconds=60.0,  # 1 minute
        )

        cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.SYSTEM_NATIVE
        )
        assert cost == 0.0  # System native should be free

    def test_cost_estimation_cloud_services(self):
        """Test cost estimation for cloud services."""
        audio_input = AudioInput(
            audio_data=None,
            audio_file_path="/path/to/audio.wav",
            sample_rate=44100,
            channels=1,
            bit_depth=16,
            duration_seconds=120.0,  # 2 minutes
        )

        # Test different cloud services
        openai_cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.OPENAI_WHISPER
        )
        google_cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.GOOGLE_SPEECH
        )
        azure_cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.AZURE_SPEECH
        )

        assert openai_cost == 2.0 * 0.006  # 2 minutes * $0.006 per minute
        assert google_cost == 2.0 * 0.004  # 2 minutes * $0.004 per minute
        assert azure_cost == 2.0 * 0.003  # 2 minutes * $0.003 per minute

    def test_cost_estimation_duration_calculation(self):
        """Test duration calculation from audio data."""
        # 16000 Hz, 1 channel, 16 bits = 2 bytes per sample
        # 10 seconds of audio = 16000 * 10 * 2 = 320,000 bytes
        audio_data_10_seconds = b"x" * 320000

        audio_input = AudioInput(
            audio_data=audio_data_10_seconds,
            audio_file_path=None,
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_seconds=None,  # Let it calculate from data size
        )

        cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.OPENAI_WHISPER
        )
        expected_duration_minutes = 10.0 / 60.0  # 10 seconds in minutes
        expected_cost = expected_duration_minutes * 0.006

        # Allow small floating point differences
        assert abs(cost - expected_cost) < 0.001

    def test_cost_estimation_default_duration(self):
        """Test cost estimation with default duration fallback."""
        audio_input = AudioInput(
            audio_data=None,
            audio_file_path="/path/to/audio.wav",  # File path provided but no duration
            sample_rate=16000,
            channels=1,
            bit_depth=16,
            duration_seconds=None,  # No duration to trigger default fallback
        )

        cost = estimate_recognition_cost(
            audio_input, SpeechRecognitionEngine.OPENAI_WHISPER
        )
        expected_cost = (30.0 / 60.0) * 0.006  # 30 seconds default * rate

        # Allow small floating point differences
        assert abs(cost - expected_cost) < 0.001


class TestPropertyBasedVoiceArchitecture:
    """Property-based tests for voice architecture."""

    @given(
        personalization=st.floats(min_value=0.0, max_value=1.0),
        confidence=st.floats(min_value=0.0, max_value=1.0),
    )
    def test_voice_profile_personalization_bounds(
        self, personalization: float, confidence: float
    ):
        """Property: Voice profiles should accept any valid personalization level."""
        now = datetime.now(UTC)

        profile = VoiceProfile(
            profile_id="test_profile",
            user_name="Test User",
            acoustic_characteristics={},
            personalization_level=personalization,
            supported_languages=[VoiceLanguage.ENGLISH_US],
            created_date=now,
            last_updated=now,
        )

        assert 0.0 <= profile.personalization_level <= 1.0

    @given(
        sample_rate=st.integers(min_value=8000, max_value=48000),
        channels=st.integers(min_value=1, max_value=2),
        bit_depth=st.integers(min_value=8, max_value=32),
    )
    def test_audio_input_parameter_ranges(
        self, sample_rate: int, channels: int, bit_depth: int
    ):
        """Property: Audio input should accept valid technical parameters."""
        audio_input = AudioInput(
            audio_data=b"test_audio_data",
            audio_file_path=None,
            sample_rate=sample_rate,
            channels=channels,
            bit_depth=bit_depth,
        )

        info = audio_input.get_audio_info()
        assert info["sample_rate"] == sample_rate
        assert info["channels"] == channels
        assert info["bit_depth"] == bit_depth

    @given(confidence_threshold=st.floats(min_value=0.0, max_value=1.0))
    def test_recognition_confidence_threshold_bounds(self, confidence_threshold: float):
        """Property: Recognition settings should accept any valid confidence threshold."""
        settings = RecognitionSettings(
            engine=SpeechRecognitionEngine.SYSTEM_NATIVE,
            language=VoiceLanguage.ENGLISH_US,
            confidence_threshold=confidence_threshold,
        )

        assert 0.0 <= settings.confidence_threshold <= 1.0

    @given(
        user_name=st.text(min_size=1, max_size=100),
        profile_id=st.text(min_size=1, max_size=50),
    )
    def test_speaker_id_generation_consistency(self, user_name: str, profile_id: str):
        """Property: Speaker ID generation should be consistent and safe."""
        assume(user_name.strip())  # Ensure non-empty after stripping
        assume(profile_id.strip())  # Ensure non-empty after stripping

        speaker_id = create_speaker_id(user_name)

        # Should always start with speaker_
        assert speaker_id.startswith("speaker_")

        # Should only contain safe characters
        assert all(c.isalnum() or c == "_" for c in speaker_id)

        # Should be reproducible for same input (at least the prefix part)
        speaker_id2 = create_speaker_id(user_name)
        name_part1 = speaker_id.split("_")[1]
        name_part2 = speaker_id2.split("_")[1]
        assert name_part1 == name_part2  # Name processing should be consistent
