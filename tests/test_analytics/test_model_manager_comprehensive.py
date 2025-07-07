"""Comprehensive tests for Model Manager with systematic coverage.

Tests cover ModelConfiguration, ModelMetadata, model lifecycle management,
training pipelines, validation frameworks, and enterprise-grade ML operations.
"""

from __future__ import annotations

import os
import tempfile
from datetime import UTC, datetime
from typing import Any
from unittest.mock import Mock, patch

import pytest
from hypothesis import given
from hypothesis import strategies as st
from src.analytics.model_manager import (
    DeploymentInfo,
    ModelArtifact,
    ModelCategory,
    ModelConfiguration,
    ModelManager,
    ModelStatus,
    TrainingDataset,
    ValidationMethod,
    ValidationResult,
)
from src.core.predictive_modeling import (
    ModelPerformance,
    ModelTrainingError,
    ModelType,
    create_model_id,
)


# Test data generators
@st.composite
def model_status_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid model statuses."""
    return draw(st.sampled_from(list(ModelStatus)))


@st.composite
def model_category_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid model categories."""
    return draw(st.sampled_from(list(ModelCategory)))


@st.composite
def model_type_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid model types."""
    return draw(
        st.sampled_from(
            [
                ModelType.LINEAR_REGRESSION,
                ModelType.RANDOM_FOREST,
                ModelType.GRADIENT_BOOSTING,
                ModelType.LSTM,
            ],
        ),
    )


@st.composite
def validation_method_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid validation methods."""
    return draw(st.sampled_from(list(ValidationMethod)))


@st.composite
def feature_columns_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid feature column lists."""
    return draw(
        st.lists(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=["Lu", "Ll"]),
            ),
            min_size=1,
            max_size=10,
            unique=True,
        ),
    )


@st.composite
def hyperparameters_strategy(draw: Callable[..., Any]) -> Any:
    """Generate valid hyperparameter dictionaries."""
    return draw(
        st.dictionaries(
            st.text(
                min_size=1,
                max_size=20,
                alphabet=st.characters(whitelist_categories=["Lu", "Ll"]),
            ),
            st.one_of(
                st.integers(min_value=1, max_value=1000),
                st.floats(
                    min_value=0.001,
                    max_value=1.0,
                    allow_nan=False,
                    allow_infinity=False,
                ),
                st.text(min_size=1, max_size=50),
            ),
            min_size=0,
            max_size=5,
        ),
    )


class TestModelStatus:
    """Test ModelStatus enum and related functionality."""

    def test_model_status_enum_values(self) -> None:
        """Test ModelStatus enum has expected values."""
        assert ModelStatus.CREATED.value == "created"
        assert ModelStatus.TRAINING.value == "training"
        assert ModelStatus.TRAINED.value == "trained"
        assert ModelStatus.VALIDATING.value == "validating"
        assert ModelStatus.VALIDATED.value == "validated"
        assert ModelStatus.DEPLOYED.value == "deployed"
        assert ModelStatus.DEPRECATED.value == "deprecated"
        assert ModelStatus.FAILED.value == "failed"

    def test_model_status_enumeration(self) -> None:
        """Test ModelStatus enum can be enumerated."""
        statuses = list(ModelStatus)
        assert len(statuses) == 8

        status_values = [status.value for status in statuses]
        expected_values = [
            "created",
            "training",
            "trained",
            "validating",
            "validated",
            "deployed",
            "deprecated",
            "failed",
        ]

        for expected in expected_values:
            assert expected in status_values


class TestModelCategory:
    """Test ModelCategory enum and related functionality."""

    def test_model_category_enum_values(self) -> None:
        """Test ModelCategory enum has expected values."""
        assert ModelCategory.TIME_SERIES_FORECASTING.value == "time_series_forecasting"
        assert ModelCategory.CLASSIFICATION.value == "classification"
        assert ModelCategory.REGRESSION.value == "regression"
        assert ModelCategory.ANOMALY_DETECTION.value == "anomaly_detection"
        assert ModelCategory.CLUSTERING.value == "clustering"
        assert ModelCategory.PATTERN_RECOGNITION.value == "pattern_recognition"

    def test_model_category_enumeration(self) -> None:
        """Test ModelCategory enum can be enumerated."""
        categories = list(ModelCategory)
        assert len(categories) == 6

        category_values = [cat.value for cat in categories]
        expected_values = [
            "time_series_forecasting",
            "classification",
            "regression",
            "anomaly_detection",
            "clustering",
            "pattern_recognition",
        ]

        for expected in expected_values:
            assert expected in category_values


class TestValidationMethod:
    """Test ValidationMethod enum and related functionality."""

    def test_validation_method_enum_values(self) -> None:
        """Test ValidationMethod enum has expected values."""
        assert ValidationMethod.TRAIN_TEST_SPLIT.value == "train_test_split"
        assert ValidationMethod.CROSS_VALIDATION.value == "cross_validation"
        assert ValidationMethod.TIME_SERIES_SPLIT.value == "time_series_split"
        assert ValidationMethod.HOLDOUT_VALIDATION.value == "holdout_validation"

    def test_validation_method_enumeration(self) -> None:
        """Test ValidationMethod enum can be enumerated."""
        methods = list(ValidationMethod)
        assert len(methods) == 4

        method_values = [method.value for method in methods]
        expected_values = [
            "train_test_split",
            "cross_validation",
            "time_series_split",
            "holdout_validation",
        ]

        for expected in expected_values:
            assert expected in method_values


class TestModelConfiguration:
    """Test ModelConfiguration with comprehensive validation."""

    def test_model_configuration_creation_valid(self) -> None:
        """Test creating valid ModelConfiguration instances."""
        model_id = create_model_id()
        config = ModelConfiguration(
            model_id=model_id,
            model_type=ModelType.RANDOM_FOREST,
            model_category=ModelCategory.CLASSIFICATION,
            target_variable="target",
            feature_columns=["feature1", "feature2", "feature3"],
            hyperparameters={"n_estimators": 100, "max_depth": 10},
            training_config={"test_size": 0.2, "random_state": 42},
            validation_config={"cv_folds": 5},
            deployment_config={"auto_deploy": True},
        )

        assert config.model_id == model_id
        assert config.model_type == ModelType.RANDOM_FOREST
        assert config.model_category == ModelCategory.CLASSIFICATION
        assert config.target_variable == "target"
        assert config.feature_columns == ["feature1", "feature2", "feature3"]
        assert config.hyperparameters["n_estimators"] == 100
        assert config.training_config["test_size"] == 0.2
        assert config.validation_config["cv_folds"] == 5
        assert config.deployment_config["auto_deploy"] is True

    def test_model_configuration_empty_target_variable(self) -> None:
        """Test ModelConfiguration with empty target variable raises ValueError."""
        model_id = create_model_id()

        with pytest.raises(ValueError, match="Target variable must be specified"):
            ModelConfiguration(
                model_id=model_id,
                model_type=ModelType.LINEAR_REGRESSION,
                model_category=ModelCategory.REGRESSION,
                target_variable="",
                feature_columns=["feature1"],
            )

    def test_model_configuration_empty_feature_columns(self) -> None:
        """Test ModelConfiguration with empty feature columns raises ValueError."""
        model_id = create_model_id()

        with pytest.raises(ValueError, match="Feature columns must be specified"):
            ModelConfiguration(
                model_id=model_id,
                model_type=ModelType.LINEAR_REGRESSION,
                model_category=ModelCategory.REGRESSION,
                target_variable="target",
                feature_columns=[],
            )

    @given(
        model_type_strategy(),
        model_category_strategy(),
        feature_columns_strategy(),
        hyperparameters_strategy(),
    )
    def test_model_configuration_property_based_creation(
        self,
        model_type: str,
        model_category: Any,
        feature_columns: Any,
        hyperparameters: Any,
    ) -> None:
        """Property-based test for ModelConfiguration creation."""
        model_id = create_model_id()
        target_variable = "target_var"

        # Ensure target is not in features
        if target_variable in feature_columns:
            feature_columns = [f for f in feature_columns if f != target_variable]

        # Ensure we have at least one feature
        if not feature_columns:
            feature_columns = ["default_feature"]

        config = ModelConfiguration(
            model_id=model_id,
            model_type=model_type,
            model_category=model_category,
            target_variable=target_variable,
            feature_columns=feature_columns,
            hyperparameters=hyperparameters,
        )

        assert config.model_id == model_id
        assert config.model_type == model_type
        assert config.model_category == model_category
        assert config.target_variable == target_variable
        assert config.feature_columns == feature_columns
        assert config.hyperparameters == hyperparameters


class TestTrainingDataset:
    """Test TrainingDataset creation and validation."""

    def test_training_dataset_creation_valid(self) -> None:
        """Test creating valid TrainingDataset instances."""
        dataset = TrainingDataset(
            dataset_id="dataset_001",
            feature_data={"feature1": [1, 2, 3], "feature2": [4, 5, 6]},
            target_data=[1, 0, 1],
            timestamps=[datetime.now(UTC) for _ in range(3)],
            metadata={"source": "test", "version": "1.0"},
            train_split=0.8,
            validation_split=0.1,
            test_split=0.1,
        )

        assert dataset.dataset_id == "dataset_001"
        assert dataset.feature_data["feature1"] == [1, 2, 3]
        assert dataset.target_data == [1, 0, 1]
        assert len(dataset.timestamps) == 3
        assert dataset.metadata["source"] == "test"
        assert dataset.train_split == 0.8
        assert dataset.validation_split == 0.1
        assert dataset.test_split == 0.1

    def test_training_dataset_empty_feature_data(self) -> None:
        """Test TrainingDataset with empty feature data raises ValueError."""
        with pytest.raises(ValueError, match="Feature data must be provided"):
            TrainingDataset(
                dataset_id="dataset_001",
                feature_data={},
                target_data=[1, 0, 1],
            )

    def test_training_dataset_empty_target_data(self) -> None:
        """Test TrainingDataset with empty target data raises ValueError."""
        with pytest.raises(ValueError, match="Target data must be provided"):
            TrainingDataset(
                dataset_id="dataset_001",
                feature_data={"feature1": [1, 2, 3]},
                target_data=[],
            )


class TestModelArtifact:
    """Test ModelArtifact creation and validation."""

    def test_model_artifact_creation_valid(self) -> None:
        """Test creating valid ModelArtifact instances."""
        model_id = create_model_id()

        # Create a temporary file for the test
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pkl") as temp_file:
            temp_file.write(b"dummy model data")
            temp_file_path = temp_file.name

        try:
            artifact = ModelArtifact(
                model_id=model_id,
                model_file_path=temp_file_path,
                model_type=ModelType.RANDOM_FOREST,
                model_category=ModelCategory.CLASSIFICATION,
                version="1.0.0",
                created_at=datetime.now(UTC),
                file_size_bytes=1024,
                checksum="abc123def456",
                metadata={"accuracy": 0.95, "training_time": 300},
            )

            assert artifact.model_id == model_id
            assert artifact.model_file_path == temp_file_path
            assert artifact.model_type == ModelType.RANDOM_FOREST
            assert artifact.model_category == ModelCategory.CLASSIFICATION
            assert artifact.version == "1.0.0"
            assert artifact.file_size_bytes == 1024
            assert artifact.checksum == "abc123def456"
            assert artifact.metadata["accuracy"] == 0.95
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_file_path):
                os.unlink(temp_file_path)


class TestValidationResult:
    """Test ValidationResult creation and validation."""

    def test_validation_result_creation_valid(self) -> None:
        """Test creating valid ValidationResult instances."""
        model_id = create_model_id()
        performance = ModelPerformance(
            model_id=model_id,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            rmse=0.15,
            mae=0.12,
            training_time_seconds=300.0,
            inference_time_ms=50.0,
        )

        result = ValidationResult(
            model_id=model_id,
            validation_method=ValidationMethod.CROSS_VALIDATION,
            performance_metrics=performance,
            validation_score=0.93,
            validation_details={"cv_folds": 5, "std_dev": 0.02},
            validation_timestamp=datetime.now(UTC),
            is_passed=True,
            failure_reasons=[],
        )

        assert result.model_id == model_id
        assert result.validation_method == ValidationMethod.CROSS_VALIDATION
        assert result.performance_metrics == performance
        assert result.validation_score == 0.93
        assert result.validation_details["cv_folds"] == 5
        assert result.is_passed is True
        assert len(result.failure_reasons) == 0

    def test_validation_result_invalid_score(self) -> None:
        """Test ValidationResult with invalid validation score raises ValueError."""
        model_id = create_model_id()
        performance = ModelPerformance(
            model_id=model_id,
            accuracy=0.95,
            precision=0.92,
            recall=0.88,
            f1_score=0.90,
            rmse=0.15,
            mae=0.12,
            training_time_seconds=300.0,
            inference_time_ms=50.0,
        )

        with pytest.raises(
            ValueError,
            match="Validation score must be between 0.0 and 1.0",
        ):
            ValidationResult(
                model_id=model_id,
                validation_method=ValidationMethod.CROSS_VALIDATION,
                performance_metrics=performance,
                validation_score=1.5,  # Invalid score
                validation_details={},
            )


class TestDeploymentInfo:
    """Test DeploymentInfo creation and validation."""

    def test_deployment_info_creation_valid(self) -> None:
        """Test creating valid DeploymentInfo instances."""
        model_id = create_model_id()
        deployment = DeploymentInfo(
            model_id=model_id,
            deployment_id="deployment_001",
            deployment_timestamp=datetime.now(UTC),
            deployment_environment="production",
            endpoint_url="https://api.example.com/predict",
            deployment_config={"replicas": 3, "memory": "2Gi"},
            health_check_url="https://api.example.com/health",
            monitoring_enabled=True,
            auto_scaling_enabled=True,
        )

        assert deployment.model_id == model_id
        assert deployment.deployment_id == "deployment_001"
        assert deployment.deployment_environment == "production"
        assert deployment.endpoint_url == "https://api.example.com/predict"
        assert deployment.deployment_config["replicas"] == 3
        assert deployment.health_check_url == "https://api.example.com/health"
        assert deployment.monitoring_enabled is True
        assert deployment.auto_scaling_enabled is True

    def test_deployment_info_invalid_environment(self) -> None:
        """Test DeploymentInfo with invalid environment raises ValueError."""
        model_id = create_model_id()

        with pytest.raises(
            ValueError,
            match="Deployment environment must be development, staging, or production",
        ):
            DeploymentInfo(
                model_id=model_id,
                deployment_id="deployment_001",
                deployment_timestamp=datetime.now(UTC),
                deployment_environment="invalid_env",
            )


class TestModelManager:
    """Test ModelManager functionality."""

    def setup_method(self) -> None:
        """Set up test fixtures."""
        self.temp_dir = tempfile.mkdtemp()
        self.manager = ModelManager(model_storage_path=self.temp_dir)

    def test_model_manager_initialization(self) -> None:
        """Test ModelManager initialization."""
        manager = ModelManager()

        assert manager.model_storage_path is not None
        assert isinstance(manager.model_registry, dict)
        assert len(manager.model_registry) == 0
        assert isinstance(manager.model_artifacts, dict)
        assert len(manager.model_artifacts) == 0
        assert isinstance(manager.model_performance, dict)
        assert len(manager.model_performance) == 0

    @pytest.mark.asyncio
    async def test_model_manager_create_model_success(self) -> None:
        """Test successful model creation."""
        model_id = create_model_id()
        config = ModelConfiguration(
            model_id=model_id,
            model_type=ModelType.RANDOM_FOREST,
            model_category=ModelCategory.CLASSIFICATION,
            target_variable="target",
            feature_columns=["feature1", "feature2"],
        )

        result = await self.manager.create_model(config)

        assert result.is_right()
        created_model_id = result.get_right()

        assert created_model_id == model_id
        assert model_id in self.manager.model_registry
        assert self.manager.model_registry[model_id] == config

    @pytest.mark.asyncio
    async def test_model_manager_create_model_unsupported_type(self) -> None:
        """Test creating model with unsupported type fails."""
        model_id = create_model_id()

        # Create a mock unsupported model type
        with patch("src.analytics.model_manager.ModelType"):
            mock_unsupported_type = Mock()
            mock_unsupported_type.value = "unsupported_type"

            config = ModelConfiguration(
                model_id=model_id,
                model_type=mock_unsupported_type,
                model_category=ModelCategory.CLASSIFICATION,
                target_variable="target",
                feature_columns=["feature1", "feature2"],
            )

            result = await self.manager.create_model(config)

            assert result.is_left()
            error = result.get_left()
            assert isinstance(error, ModelTrainingError)
            assert "Unsupported model type" in str(error)

    @pytest.mark.asyncio
    async def test_model_manager_train_model_success(self) -> None:
        """Test successful model training."""
        model_id = create_model_id()
        config = ModelConfiguration(
            model_id=model_id,
            model_type=ModelType.LINEAR_REGRESSION,
            model_category=ModelCategory.REGRESSION,
            target_variable="target",
            feature_columns=["feature1", "feature2"],
        )

        # Create model first
        create_result = await self.manager.create_model(config)
        assert create_result.is_right()

        # Create training dataset
        dataset = TrainingDataset(
            dataset_id="dataset_001",
            feature_data={
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20],
            },
            target_data=[3, 6, 9, 12, 15, 18, 21, 24, 27, 30],
        )

        # Mock the training implementation methods
        with (
            patch.object(self.manager, "_prepare_training_data") as mock_prepare,
            patch.object(self.manager, "_execute_training") as mock_execute,
            patch.object(self.manager, "_validate_model") as mock_validate,
            patch.object(self.manager, "_save_model_artifact") as mock_save,
        ):
            # Mock successful data preparation - return Either directly
            from src.core.either import Either

            mock_prepare.return_value = Either.right(
                (
                    {"feature1": [1, 2, 3], "feature2": [4, 5, 6]},  # train
                    {"feature1": [7, 8], "feature2": [9, 10]},  # val
                    {"feature1": [9, 10], "feature2": [11, 12]},  # test
                ),
            )

            # Mock successful training - return Either directly
            mock_execute.return_value = Either.right({"trained_model": "mock_model"})

            # Mock successful validation - return ValidationResult wrapped in Either
            from src.analytics.model_manager import ValidationMethod, ValidationResult

            mock_performance = ModelPerformance(
                model_id=model_id,
                accuracy=0.95,
                precision=0.92,
                recall=0.88,
                f1_score=0.90,
                rmse=0.15,
                mae=0.12,
                training_time_seconds=300.0,
                inference_time_ms=50.0,
            )
            mock_validation_result = ValidationResult(
                model_id=model_id,
                validation_method=ValidationMethod.TRAIN_TEST_SPLIT,
                performance_metrics=mock_performance,
                validation_score=0.93,
                validation_details={
                    "precision": 0.92,
                    "recall": 0.88,
                    "f1_score": 0.90,
                    "rmse": 0.15,
                    "mae": 0.12,
                    "inference_time_ms": 50.0,
                },
            )
            mock_validate.return_value = Either.right(mock_validation_result)

            # Mock successful artifact saving - return Either directly
            mock_save.return_value = Either.right(None)

            result = await self.manager.train_model(model_id, dataset)

            assert result.is_right()
            performance = result.get_right()
            assert isinstance(performance, ModelPerformance)
            assert performance.model_id == model_id

    @pytest.mark.asyncio
    async def test_model_manager_train_unregistered_model(self) -> None:
        """Test training unregistered model fails."""
        model_id = create_model_id()
        dataset = TrainingDataset(
            dataset_id="dataset_001",
            feature_data={
                "feature1": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
                "feature2": [2, 4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24],
            },
            target_data=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        )

        result = await self.manager.train_model(model_id, dataset)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, ModelTrainingError)
        assert "not found in registry" in str(error)

    @pytest.mark.asyncio
    async def test_model_manager_get_model_status(self) -> None:
        """Test getting model status."""
        model_id = create_model_id()
        config = ModelConfiguration(
            model_id=model_id,
            model_type=ModelType.RANDOM_FOREST,
            model_category=ModelCategory.CLASSIFICATION,
            target_variable="target",
            feature_columns=["feature1", "feature2"],
        )

        # Create model
        create_result = await self.manager.create_model(config)
        assert create_result.is_right()

        # Get status
        status = await self.manager.get_model_status(model_id)

        assert isinstance(status, dict)
        assert status["model_id"] == model_id
        assert status["status"] == ModelStatus.CREATED.value
        assert status["model_type"] == ModelType.RANDOM_FOREST.value
        assert status["model_category"] == ModelCategory.CLASSIFICATION.value

    @pytest.mark.asyncio
    async def test_model_manager_get_nonexistent_model_status(self) -> None:
        """Test getting status for nonexistent model."""
        model_id = create_model_id()

        status = await self.manager.get_model_status(model_id)

        assert isinstance(status, dict)
        assert "error" in status
        assert "not found" in status["error"]

    @pytest.mark.asyncio
    async def test_model_manager_get_summary(self) -> None:
        """Test getting model manager summary."""
        # Create a few models
        for i in range(3):
            model_id = create_model_id()
            config = ModelConfiguration(
                model_id=model_id,
                model_type=ModelType.LINEAR_REGRESSION,
                model_category=ModelCategory.REGRESSION,
                target_variable=f"target_{i}",
                feature_columns=[f"feature_{i}_1", f"feature_{i}_2"],
            )

            result = await self.manager.create_model(config)
            assert result.is_right()

        summary = await self.manager.get_model_manager_summary()

        assert isinstance(summary, dict)
        assert summary["total_models"] == 3
        assert summary["trained_models"] == 0  # No models trained yet
        assert summary["deployed_models"] == 0  # No models deployed yet
        assert "model_types_distribution" in summary
        assert "average_performance" in summary
        assert "storage_info" in summary
        assert "validation_methods_supported" in summary
        assert "model_types_supported" in summary
        assert "summary_timestamp" in summary

    @pytest.mark.asyncio
    async def test_model_manager_multiple_model_types(self) -> None:
        """Test ModelManager with multiple different model types."""
        model_configs = [
            (ModelType.LINEAR_REGRESSION, ModelCategory.REGRESSION),
            (ModelType.RANDOM_FOREST, ModelCategory.CLASSIFICATION),
            (ModelType.LSTM, ModelCategory.TIME_SERIES_FORECASTING),
            (ModelType.GRADIENT_BOOSTING, ModelCategory.CLASSIFICATION),
        ]

        created_models = []

        for model_type, category in model_configs:
            model_id = create_model_id()
            config = ModelConfiguration(
                model_id=model_id,
                model_type=model_type,
                model_category=category,
                target_variable="target",
                feature_columns=["feature1", "feature2"],
            )

            result = await self.manager.create_model(config)
            assert result.is_right()
            created_models.append(model_id)

        # Verify all models are created
        summary = await self.manager.get_model_manager_summary()
        assert summary["total_models"] == 4

        # Verify we can get status for each model
        for model_id in created_models:
            status = await self.manager.get_model_status(model_id)
            assert "error" not in status
            assert status["model_id"] == model_id
            assert status["status"] == ModelStatus.CREATED.value
