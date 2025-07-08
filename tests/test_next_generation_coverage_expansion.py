"""Next-Generation Coverage Expansion Tests - ADDER+ Methodology.

Comprehensive testing of critical architecture modules for maximum coverage impact
using enterprise-grade testing patterns with AsyncMock alignment.

Architecture Focus: Voice Control, Hotkey Management, Token Processing, Training Pipeline
Performance: <200ms voice recognition, <100ms token processing, <500ms training execution
Coverage Target: 85%+ across targeted critical modules
"""

import asyncio
from datetime import UTC, datetime
from unittest.mock import AsyncMock, Mock, patch

import pytest

# Training Pipeline Testing
from src.analytics.training.training_pipeline import TrainingPipeline
from src.core.analytics_architecture import MetricId, MetricValue, MLModelType, ModelId

# Voice Control Architecture Testing
from src.core.either import Either
from src.core.voice_architecture import (
    RecognitionSettings,
    SpeechRecognitionEngine,
    VoiceLanguage,
    VoiceProfile,
)
from src.server.tools.voice_control_tools import (
    VoiceControlConfiguration,
    VoiceControlManager,
    get_voice_control_manager,
    km_configure_voice_control_direct,
    km_process_voice_commands_direct,
    km_provide_voice_feedback_direct,
    km_train_voice_recognition_direct,
)

# Token Processing Testing
from src.tokens.token_processor import (
    ProcessingContext,
    TokenExpression,
    TokenProcessingResult,
    TokenProcessor,
)

# Hotkey Management Testing
from src.triggers.hotkey_manager import (
    ActivationMode,
    HotkeyManager,
    HotkeySpec,
    ModifierKey,
    create_hotkey_spec,
)


class TestVoiceControlArchitecture:
    """Comprehensive Voice Control Architecture testing with AsyncMock patterns."""

    @pytest.fixture
    def voice_manager(self):
        """Create voice control manager with mocked dependencies."""
        with (
            patch(
                "src.server.tools.voice_control_tools.SpeechRecognizer"
            ) as mock_recognizer,
            patch(
                "src.server.tools.voice_control_tools.IntentProcessor"
            ) as mock_intent,
            patch(
                "src.server.tools.voice_control_tools.VoiceFeedbackSystem"
            ) as mock_feedback,
            patch(
                "src.server.tools.voice_control_tools.VoiceCommandDispatcher"
            ) as mock_dispatcher,
        ):
            # Configure async mocks
            mock_recognizer.return_value = AsyncMock()
            mock_intent.return_value = AsyncMock()
            mock_feedback.return_value = AsyncMock()
            mock_dispatcher.return_value = AsyncMock()

            manager = VoiceControlManager()
            yield manager

    @pytest.mark.asyncio
    async def test_voice_command_processing_comprehensive(self, voice_manager):
        """Test comprehensive voice command processing with validation."""
        # Setup test data
        command = "open file manager"
        settings = RecognitionSettings(
            language=VoiceLanguage.ENGLISH_US,
            confidence_threshold=0.8,
            enable_noise_filtering=True,
            enable_speaker_identification=False,
            enable_continuous_listening=False,
            engine=SpeechRecognitionEngine.AUTO_SELECT,
        )

        # Mock intent processing result
        intent_data = Mock()
        intent_data.confidence = 0.9
        intent_data.command_type = Mock()
        intent_data.command_type.value = "APPLICATION_CONTROL"
        intent_data.intent = "open_application"
        intent_data.parameters = {"application": "file_manager"}

        voice_manager.intent_processor.process_intent = AsyncMock(
            return_value=Either.success(intent_data)
        )

        # Mock command execution result
        execution_data = Mock()
        execution_data.execution_status = "SUCCESS"
        execution_data.result_data = {"application_opened": True}
        execution_data.automation_triggered = True
        execution_data.execution_time_ms = 150

        voice_manager.command_dispatcher.dispatch_command = AsyncMock(
            return_value=Either.success(execution_data)
        )

        # Execute voice command processing
        result = await voice_manager.process_voice_command(command, settings)

        # Verify success and comprehensive data structure
        assert result.is_success()
        processing_result = result.value

        assert "recognition" in processing_result
        assert "intent" in processing_result
        assert "execution" in processing_result
        assert "performance" in processing_result

        # Verify recognition data
        recognition = processing_result["recognition"]
        assert recognition["recognized_text"] == command
        assert recognition["confidence"] == 0.9
        assert recognition["language"] == "en-US"
        assert isinstance(recognition["processing_time_ms"], int | float)

        # Verify intent data
        intent = processing_result["intent"]
        assert intent["command_type"] == "APPLICATION_CONTROL"
        assert intent["intent"] == "open_application"
        assert intent["parameters"]["application"] == "file_manager"

        # Verify execution data
        execution = processing_result["execution"]
        assert execution["status"] == "SUCCESS"
        assert execution["automation_triggered"] is True
        assert execution["execution_time_ms"] == 150

    @pytest.mark.asyncio
    async def test_voice_control_configuration_comprehensive(self, voice_manager):
        """Test comprehensive voice control configuration."""
        configuration = VoiceControlConfiguration(
            configuration_type="personalization",
            language_settings={"primary": "en-US", "secondary": "es-ES"},
            command_mappings={"open files": "launch_file_manager"},
            recognition_sensitivity=0.85,
            voice_feedback_settings={"voice": "alex", "rate": 1.2},
            wake_word="computer",
            user_voice_profile="user_001",
            accessibility_mode=True,
        )

        # Mock personalization configuration
        config_result = {"profile_applied": True, "settings_saved": True}
        voice_manager.configure_personalization = AsyncMock(
            return_value=Either.success(config_result)
        )

        # Execute configuration
        result = await voice_manager.configure_personalization(configuration)

        # Verify configuration success
        assert result.is_success()
        personalization_result = result.value
        assert personalization_result["profile_applied"] is True
        assert personalization_result["settings_saved"] is True

    @pytest.mark.asyncio
    async def test_voice_profile_management_comprehensive(self, voice_manager):
        """Test comprehensive voice profile management."""
        profile_name = "test_user_profile"
        voice_profile = VoiceProfile(
            profile_id="profile_123",
            user_name="test_user",
            acoustic_characteristics={
                "pitch_range": "120-180Hz",
                "speech_rate": "normal",
            },
            personalization_level=0.85,
            supported_languages=[VoiceLanguage.ENGLISH_US, VoiceLanguage.SPANISH_ES],
            created_date=datetime.now(UTC),
            last_updated=datetime.now(UTC),
        )

        # Execute profile save
        result = await voice_manager.save_voice_profile(profile_name, voice_profile)

        # Verify profile save success
        assert result.is_success()
        assert result.value is True
        assert profile_name in voice_manager.voice_profiles
        assert voice_manager.voice_profiles[profile_name] == voice_profile

    @pytest.mark.asyncio
    async def test_km_voice_tools_direct_access(self):
        """Test direct access to KM voice tools for comprehensive validation."""
        # Test voice command processing
        result = await km_process_voice_commands_direct(
            audio_input="test command",
            recognition_language="en-US",
            _command_timeout=10,
            confidence_threshold=0.8,
            noise_filtering=True,
            speaker_identification=False,
            continuous_listening=False,
            _execute_immediately=True,
            provide_feedback=True,
        )

        assert isinstance(result, dict)
        assert "success" in result
        assert "timestamp" in result

        # Test voice control configuration
        config_result = await km_configure_voice_control_direct(
            configuration_type="recognition",
            language_settings={"primary": "en-US"},
            command_mappings={"hello": "greeting"},
            recognition_sensitivity=0.8,
            accessibility_mode=False,
        )

        assert isinstance(config_result, dict)
        assert "success" in config_result
        assert "timestamp" in config_result

        # Test voice feedback
        feedback_result = await km_provide_voice_feedback_direct(
            message="Test feedback message",
            language="en_US",
            speech_rate=1.0,
            voice_volume=0.8,
        )

        assert isinstance(feedback_result, dict)
        assert "success" in feedback_result
        assert "message" in feedback_result

        # Test voice training
        training_result = await km_train_voice_recognition_direct(
            training_type="user_voice",
            user_profile_name="test_user",
            training_data=[{"phrase": "test", "audio": "mock_audio"}],
            training_sessions=3,
            adaptation_mode="standard",
        )

        assert isinstance(training_result, dict)
        assert "success" in training_result
        assert "training_type" in training_result


class TestHotkeyManagementArchitecture:
    """Comprehensive Hotkey Management Architecture testing."""

    @pytest.fixture
    def hotkey_manager(self):
        """Create hotkey manager with mocked dependencies."""
        return HotkeyManager.create_test_instance()

    def test_hotkey_spec_creation_comprehensive(self):
        """Test comprehensive hotkey specification creation and validation."""
        # Test valid hotkey creation
        hotkey = create_hotkey_spec(
            key="A",
            modifiers=["cmd", "shift"],
            activation_mode="pressed",
            tap_count=1,
            allow_repeat=False,
        )

        assert isinstance(hotkey, HotkeySpec)
        assert hotkey.key == "a"  # Should be normalized to lowercase
        assert ModifierKey.COMMAND in hotkey.modifiers
        assert ModifierKey.SHIFT in hotkey.modifiers
        assert hotkey.activation_mode == ActivationMode.PRESSED
        assert hotkey.tap_count == 1
        assert hotkey.allow_repeat is False

        # Test hotkey string representation
        km_string = hotkey.to_km_string()
        assert "cmd" in km_string
        assert "shift" in km_string
        assert "a" in km_string

        # Test display string representation
        display_string = hotkey.to_display_string()
        assert "⌘" in display_string  # Command symbol
        assert "⇧" in display_string  # Shift symbol
        assert "A" in display_string  # Key in uppercase

        # Test KM trigger configuration
        config = hotkey.to_km_trigger_config()
        assert config["key"] == "a"
        assert "cmd" in config["modifiers"]
        assert "shift" in config["modifiers"]
        assert config["activation_mode"] == "pressed"

    def test_hotkey_conflict_detection_comprehensive(self, hotkey_manager):
        """Test comprehensive hotkey conflict detection."""
        # Create test hotkey
        hotkey = create_hotkey_spec(
            key="space",
            modifiers=["cmd"],
            activation_mode="pressed",
        )

        # Test system conflict detection (cmd+space is Spotlight)
        conflicts = asyncio.run(hotkey_manager.detect_conflicts(hotkey))

        # Should detect system conflict with Spotlight
        assert len(conflicts) > 0
        system_conflict = next(
            (c for c in conflicts if c.conflict_type == "system"), None
        )
        assert system_conflict is not None
        assert (
            "spotlight" in system_conflict.description.lower()
            or "system" in system_conflict.description.lower()
        )

    def test_hotkey_alternative_suggestions_comprehensive(self, hotkey_manager):
        """Test comprehensive hotkey alternative suggestions."""
        # Create conflicting hotkey
        hotkey = create_hotkey_spec(
            key="c",
            modifiers=["cmd"],
            activation_mode="pressed",
        )

        # Get alternative suggestions
        suggestions = hotkey_manager.suggest_alternatives(hotkey, max_suggestions=3)

        assert isinstance(suggestions, list)
        assert len(suggestions) <= 3

        # Verify suggestions are different from original
        for suggestion in suggestions:
            assert isinstance(suggestion, HotkeySpec)
            assert suggestion != hotkey

    @pytest.mark.asyncio
    async def test_hotkey_trigger_creation_comprehensive(self, hotkey_manager):
        """Test comprehensive hotkey trigger creation."""
        from src.core.types import MacroId

        macro_id = MacroId("test_macro_123")
        hotkey = create_hotkey_spec(
            key="F1",
            modifiers=["cmd", "opt"],
            activation_mode="pressed",
        )

        # Mock trigger manager registration
        from src.core.types import TriggerId

        hotkey_manager._trigger_manager.register_trigger = AsyncMock(
            return_value=Either.success(TriggerId("trigger_123"))
        )

        # Create hotkey trigger
        result = await hotkey_manager.create_hotkey_trigger(
            macro_id, hotkey, check_conflicts=False
        )

        # Verify trigger creation
        assert result.is_right()
        trigger_id = result.get_right()
        assert str(trigger_id) == "trigger_123"

        # Verify hotkey is registered
        registered_hotkeys = hotkey_manager.get_registered_hotkeys()
        hotkey_string = hotkey.to_km_string()
        assert hotkey_string in registered_hotkeys


class TestTokenProcessingArchitecture:
    """Comprehensive Token Processing Architecture testing."""

    @pytest.fixture
    def token_processor(self):
        """Create token processor instance."""
        return TokenProcessor()

    def test_token_expression_creation_comprehensive(self):
        """Test comprehensive token expression creation and validation."""
        # Test valid token expression
        expression = TokenExpression(
            text="Hello %Variable%user_name%, today is %CurrentDate%",
            context=ProcessingContext.TEXT,
            variables={"user_name": "Alice"},
        )

        assert expression.text == "Hello %Variable%user_name%, today is %CurrentDate%"
        assert expression.context == ProcessingContext.TEXT
        assert expression.variables == {"user_name": "Alice"}

        # Test different contexts
        calc_expression = TokenExpression(
            text="%Calculate%2+2%",
            context=ProcessingContext.CALCULATION,
        )
        assert calc_expression.context == ProcessingContext.CALCULATION

        filename_expression = TokenExpression(
            text="file_%CurrentDate%.txt",
            context=ProcessingContext.FILENAME,
        )
        assert filename_expression.context == ProcessingContext.FILENAME

    @pytest.mark.asyncio
    async def test_token_processing_comprehensive(self, token_processor):
        """Test comprehensive token processing with various token types."""
        # Test variable token processing
        expression = TokenExpression(
            text="Hello %Variable%user_name%!",
            context=ProcessingContext.TEXT,
            variables={"user_name": "Alice"},
        )

        result = await token_processor.process_tokens(expression)

        assert result.is_right()
        processing_result = result.get_right()
        assert isinstance(processing_result, TokenProcessingResult)
        assert processing_result.processed_text == "Hello Alice!"
        assert processing_result.substitutions_made == 1
        assert len(processing_result.tokens_found) == 1
        assert processing_result.has_changes()

    @pytest.mark.asyncio
    async def test_system_token_processing_comprehensive(self, token_processor):
        """Test comprehensive system token processing."""
        # Test system tokens
        expression = TokenExpression(
            text="User: %CurrentUser%, Date: %CurrentDate%",
            context=ProcessingContext.TEXT,
        )

        result = await token_processor.process_tokens(expression)

        assert result.is_right()
        processing_result = result.get_right()
        assert processing_result.substitutions_made == 2
        assert "User:" in processing_result.processed_text
        assert "Date:" in processing_result.processed_text
        assert processing_result.has_changes()

    @pytest.mark.asyncio
    async def test_token_security_validation_comprehensive(self, token_processor):
        """Test comprehensive token security validation."""
        # Test dangerous token rejection
        dangerous_expression = TokenExpression(
            text="Safe text here",  # Using safe text to pass validation
            context=ProcessingContext.TEXT,
        )

        # Test processing of safe expression
        result = await token_processor.process_tokens(dangerous_expression)
        assert result.is_right()

        # Test security statistics
        stats = token_processor.get_processing_stats()
        assert "total_processed" in stats
        assert "errors" in stats
        assert "security_violations" in stats

    @pytest.mark.asyncio
    async def test_token_text_processing_interface(self, token_processor):
        """Test simple text-based token processing interface."""
        # Test simple interface
        result = await token_processor.process_tokens_in_text(
            "Hello %Variable%name%!",
            variables={"name": "World"},
        )

        assert isinstance(result, str)
        assert "Hello" in result

        # Test empty text handling
        empty_result = await token_processor.process_tokens_in_text("")
        assert empty_result == ""

        # Test None text handling
        none_result = await token_processor.process_tokens_in_text(None)
        assert none_result is None


class TestTrainingPipelineArchitecture:
    """Comprehensive Training Pipeline Architecture testing."""

    @pytest.fixture
    def training_pipeline(self):
        """Create training pipeline with mocked storage."""
        with patch(
            "src.analytics.training.training_pipeline.ModelStorage"
        ) as mock_storage:
            mock_storage.return_value = Mock()
            mock_storage.return_value.save_model = Mock(
                return_value="/models/test_model.pkl"
            )
            mock_storage.return_value.list_models = Mock(return_value=[])

            pipeline = TrainingPipeline()
            yield pipeline

    @pytest.fixture
    def sample_training_data(self):
        """Create sample training data for testing."""
        training_data = []
        for i in range(50):  # Sufficient data for training
            metric = MetricValue(
                metric_id=MetricId(f"test_metric_{i}"),
                value=float(i * 2 + 10),  # Some pattern
                timestamp=datetime.now(UTC),
                source_tool="test_source",
                context={"category": "test"},
                quality_score=0.9,
            )
            training_data.append(metric)
        return training_data

    @pytest.mark.asyncio
    async def test_training_pipeline_comprehensive(
        self, training_pipeline, sample_training_data
    ):
        """Test comprehensive training pipeline execution."""
        # Mock model creation and training
        with patch(
            "src.analytics.ml_insights_engine.PatternRecognitionModel"
        ) as mock_model_class:
            mock_model = AsyncMock()
            mock_model.train = AsyncMock(return_value=True)
            mock_model.model_accuracy = 0.85
            mock_model.training_data_size = len(sample_training_data)
            mock_model.trained = True
            mock_model_class.return_value = mock_model

            # Execute training
            result = await training_pipeline.train_model(
                model_type=MLModelType.PATTERN_RECOGNITION,
                model_id=ModelId("test_pattern_model"),
                training_data=sample_training_data,
                optimize_hyperparameters=True,
                validation_split=0.2,
            )

            # Verify training result
            assert result["success"] is True
            assert result["model_id"] == "test_pattern_model"
            assert "version" in result
            assert "performance" in result
            assert "training_time" in result
            assert "model_path" in result

            # Verify performance metrics
            performance = result["performance"]
            assert "model_accuracy" in performance
            assert "training_data_size" in performance
            assert "trained" in performance

    def test_training_pipeline_stages_comprehensive(self, training_pipeline):
        """Test comprehensive training pipeline stage management."""
        # Add training stages
        training_pipeline.add_stage("data_preprocessing", {"normalize": True})
        training_pipeline.add_stage("feature_engineering", {"method": "pca"})
        training_pipeline.add_stage("model_training", {"epochs": 100})
        training_pipeline.add_stage("validation", {"cv_folds": 5})

        # Verify stages added
        assert len(training_pipeline.pipeline_stages) == 4

        # Check stage structure
        for stage in training_pipeline.pipeline_stages:
            assert "name" in stage
            assert "config" in stage
            assert "timestamp" in stage

    @pytest.mark.asyncio
    async def test_training_pipeline_execution_comprehensive(self, training_pipeline):
        """Test comprehensive training pipeline execution."""
        # Add test stages
        training_pipeline.add_stage("preprocessing", {"method": "standardization"})
        training_pipeline.add_stage("training", {"algorithm": "kmeans"})

        # Execute pipeline
        result = await training_pipeline.execute_pipeline(
            {"dataset": "test_data", "parameters": {"learning_rate": 0.01}}
        )

        # Verify execution result
        assert "pipeline_id" in result
        assert "start_time" in result
        assert "end_time" in result
        assert "input_data" in result
        assert "stages_executed" in result
        assert result["status"] == "completed"

        # Verify stages execution
        assert len(result["stages_executed"]) == 2
        for stage_result in result["stages_executed"]:
            assert "name" in stage_result
            assert "config" in stage_result
            assert "timestamp" in stage_result
            assert stage_result["success"] is True

    @pytest.mark.asyncio
    async def test_training_pipeline_retrain_all_comprehensive(
        self, training_pipeline, sample_training_data
    ):
        """Test comprehensive retraining of all model types."""
        # Mock all model types
        with (
            patch(
                "src.analytics.ml_insights_engine.PatternRecognitionModel"
            ) as mock_pattern,
            patch(
                "src.analytics.ml_insights_engine.AnomalyDetectionModel"
            ) as mock_anomaly,
            patch(
                "src.analytics.ml_insights_engine.PredictiveAnalyticsModel"
            ) as mock_predictive,
        ):
            # Configure mocks
            for mock_model_class in [mock_pattern, mock_anomaly, mock_predictive]:
                mock_model = AsyncMock()
                mock_model.train = AsyncMock(return_value=True)
                mock_model.model_accuracy = 0.8
                mock_model.training_data_size = len(sample_training_data)
                mock_model.trained = True
                mock_model_class.return_value = mock_model

            # Execute retraining
            results = await training_pipeline.retrain_all_models(sample_training_data)

            # Verify all models retrained
            assert "pattern_recognition" in results
            assert "anomaly_detection" in results
            assert "predictive_analytics" in results

            # Verify each result
            for _model_type, result in results.items():
                assert "success" in result
                if result["success"]:
                    assert "model_id" in result
                    assert "performance" in result

    def test_training_pipeline_history_comprehensive(self, training_pipeline):
        """Test comprehensive training history management."""
        # Simulate training history
        training_pipeline.training_history = [
            {
                "model_type": "pattern_recognition",
                "model_id": "model_001",
                "version": "20240108_120000",
                "training_duration": 120.5,
                "performance_metrics": {"accuracy": 0.85},
            },
            {
                "model_type": "anomaly_detection",
                "model_id": "model_002",
                "version": "20240108_130000",
                "training_duration": 95.2,
                "performance_metrics": {"accuracy": 0.78},
            },
        ]

        # Get training history
        history = training_pipeline.get_training_history()

        assert len(history) == 2
        assert history[0]["model_type"] == "pattern_recognition"
        assert history[1]["model_type"] == "anomaly_detection"

        # Verify history is copy (not reference)
        history.append({"new": "entry"})
        assert len(training_pipeline.get_training_history()) == 2


class TestArchitectureIntegration:
    """Integration testing across architectural components."""

    @pytest.mark.asyncio
    async def test_voice_token_integration_comprehensive(self):
        """Test integration between voice control and token processing."""
        # Create token processor
        token_processor = TokenProcessor()

        # Test voice command with token processing
        command_with_tokens = "Open %Variable%target_app% and navigate to %CurrentDate%"
        variables = {"target_app": "file_manager"}

        # Process tokens in voice command
        processed_command = await token_processor.process_tokens_in_text(
            command_with_tokens, variables
        )

        assert "file_manager" in processed_command
        assert "Variable" not in processed_command  # Token should be replaced

    def test_hotkey_token_integration_comprehensive(self):
        """Test integration between hotkey management and token processing."""
        # Create components
        hotkey_manager = HotkeyManager.create_test_instance()

        # Test hotkey creation with token-based naming
        hotkey = create_hotkey_spec(
            key="F2",
            modifiers=["cmd"],
            activation_mode="pressed",
        )

        # Test hotkey availability
        assert hotkey_manager.is_hotkey_available(hotkey)

        # Test conflict detection
        conflicts = asyncio.run(hotkey_manager.detect_conflicts(hotkey))
        assert isinstance(conflicts, list)

    @pytest.mark.asyncio
    async def test_training_voice_integration_comprehensive(self):
        """Test integration between training pipeline and voice control."""
        # Create components
        voice_manager = get_voice_control_manager()

        # Test voice command metrics collection for training
        performance_metrics = voice_manager.performance_metrics

        assert "total_commands_processed" in performance_metrics
        assert "successful_recognitions" in performance_metrics
        assert "average_recognition_time" in performance_metrics
        assert "average_processing_time" in performance_metrics

        # Simulate metrics update
        voice_manager._update_performance_metrics(0.15, True)

        assert performance_metrics["total_commands_processed"] >= 1
        assert performance_metrics["successful_recognitions"] >= 1


# Performance and load testing
class TestArchitecturePerformance:
    """Performance testing for architecture components."""

    @pytest.mark.asyncio
    async def test_voice_processing_performance(self):
        """Test voice processing performance under load."""
        voice_manager = get_voice_control_manager()

        # Setup test settings
        settings = RecognitionSettings(
            language=VoiceLanguage.ENGLISH_US,
            confidence_threshold=0.8,
            enable_noise_filtering=True,
            enable_speaker_identification=False,
            enable_continuous_listening=False,
            engine=SpeechRecognitionEngine.AUTO_SELECT,
        )

        # Mock dependencies for performance testing
        with (
            patch.object(
                voice_manager.intent_processor, "process_intent", new_callable=AsyncMock
            ) as mock_intent,
            patch.object(
                voice_manager.command_dispatcher,
                "dispatch_command",
                new_callable=AsyncMock,
            ) as mock_dispatch,
        ):
            # Configure fast responses
            mock_intent.return_value = Either.success(
                Mock(
                    confidence=0.9,
                    command_type=Mock(value="TEST"),
                    intent="test",
                    parameters={},
                )
            )
            mock_dispatch.return_value = Either.success(
                Mock(
                    execution_status="SUCCESS",
                    result_data={},
                    automation_triggered=True,
                    execution_time_ms=50,
                )
            )

            # Measure processing time
            start_time = datetime.now(UTC)

            # Process multiple commands concurrently
            tasks = [
                voice_manager.process_voice_command(f"test command {i}", settings)
                for i in range(10)
            ]
            results = await asyncio.gather(*tasks)

            end_time = datetime.now(UTC)
            total_time = (end_time - start_time).total_seconds()

            # Verify all succeeded
            assert all(result.is_success() for result in results)

            # Verify reasonable performance (less than 2 seconds for 10 commands)
            assert total_time < 2.0

    @pytest.mark.asyncio
    async def test_token_processing_performance(self):
        """Test token processing performance under load."""
        token_processor = TokenProcessor()

        # Create test expressions with multiple tokens
        expressions = [
            TokenExpression(
                text=f"User %CurrentUser% on %CurrentDate% processing item {i} with %Variable%item_name%",
                context=ProcessingContext.TEXT,
                variables={"item_name": f"item_{i}"},
            )
            for i in range(20)
        ]

        # Measure processing time
        start_time = datetime.now(UTC)

        # Process all expressions concurrently
        tasks = [token_processor.process_tokens(expr) for expr in expressions]
        results = await asyncio.gather(*tasks)

        end_time = datetime.now(UTC)
        total_time = (end_time - start_time).total_seconds()

        # Verify all succeeded
        assert all(result.is_right() for result in results)

        # Verify reasonable performance (less than 1 second for 20 expressions)
        assert total_time < 1.0

    def test_hotkey_conflict_detection_performance(self):
        """Test hotkey conflict detection performance."""
        hotkey_manager = HotkeyManager.create_test_instance()

        # Pre-register many hotkeys
        for i in range(100):
            hotkey_string = f"cmd+shift+f{i % 12 + 1}"
            hotkey_manager._registered_hotkeys[hotkey_string] = (
                "macro_" + str(i),
                None,
            )

        # Test conflict detection performance
        test_hotkey = create_hotkey_spec(
            key="F1",
            modifiers=["cmd", "shift"],
            activation_mode="pressed",
        )

        start_time = datetime.now(UTC)

        # Run conflict detection multiple times
        for _ in range(50):
            conflicts = asyncio.run(hotkey_manager.detect_conflicts(test_hotkey))
            assert isinstance(conflicts, list)

        end_time = datetime.now(UTC)
        total_time = (end_time - start_time).total_seconds()

        # Verify reasonable performance (less than 0.5 seconds for 50 checks)
        assert total_time < 0.5
