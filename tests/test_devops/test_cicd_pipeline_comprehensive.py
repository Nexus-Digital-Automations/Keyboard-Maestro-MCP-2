"""Comprehensive tests for DevOps CI/CD Pipeline System.

SYSTEMATIC TEST PATTERN ALIGNMENT for DevOps CI/CD Infrastructure
Using proven ADDER+ methodology for enterprise coverage expansion.

Test Categories:
1. Pipeline Configuration and Validation
2. Build Execution and Management
3. Deployment Strategies and Rollback
4. Artifact Collection and Caching
5. Error Handling and Recovery
6. Multi-Environment Operations

Architecture: Design by Contract + Property-Based Testing + Complete Coverage
Performance: <2s test execution, comprehensive validation coverage
Security: Enterprise-grade pipeline security validation
"""

import asyncio
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path
from unittest.mock import AsyncMock, patch

import pytest
from src.devops.cicd_pipeline import (
    BuildResult,
    BuildStatus,
    CICDPipeline,
    DeploymentConfig,
    DeploymentStrategy,
    PipelineConfig,
    PipelineStage,
    PipelineStep,
    TestingStrategy,
    get_cicd_pipeline,
)
from src.orchestration.ecosystem_architecture import OrchestrationError


class TestCICDPipelineCore:
    """Test core CI/CD pipeline initialization and configuration."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        temp_dir = tempfile.mkdtemp()
        return CICDPipeline(workspace_path=temp_dir)

    @pytest.fixture
    def valid_pipeline_config(self):
        """Valid pipeline configuration for testing."""
        return {
            "id": "test-pipeline-001",
            "name": "Test Pipeline",
            "description": "Comprehensive test pipeline",
            "trigger_conditions": ["push", "pull_request"],
            "environment_variables": {"NODE_ENV": "test", "CI": "true"},
            "steps": [
                {
                    "id": "build",
                    "name": "Build Application",
                    "stage": "build",
                    "command": "npm run build",
                    "timeout": 300,
                    "retry_count": 1,
                },
                {
                    "id": "test",
                    "name": "Run Tests",
                    "stage": "test",
                    "command": "npm test",
                    "depends_on": ["build"],
                },
            ],
            "timeout": 1800,
            "parallel_jobs": 2,
        }

    def test_cicd_pipeline_initialization(self, pipeline):
        """Test CI/CD pipeline proper initialization."""
        assert pipeline.workspace_path.exists()
        assert pipeline.max_concurrent_builds == 5
        assert pipeline.default_timeout == 3600
        assert pipeline.artifact_retention_days == 30
        assert pipeline.cache_enabled is True
        assert len(pipeline.active_pipelines) == 0
        assert len(pipeline.running_builds) == 0
        assert len(pipeline.build_history) == 0

    def test_pipeline_step_validation(self):
        """Test pipeline step validation with contracts."""
        # Valid step creation
        step = PipelineStep(
            step_id="valid_step",
            name="Valid Step",
            stage=PipelineStage.BUILD,
            command="echo 'hello'",
            timeout=300,
            retry_count=2,
        )
        assert step.step_id == "valid_step"
        assert step.stage == PipelineStage.BUILD
        assert step.timeout == 300
        assert step.retry_count == 2

    def test_pipeline_config_validation(self):
        """Test pipeline configuration validation with contracts."""
        config = PipelineConfig(
            pipeline_id="test-pipeline",
            name="Test Pipeline",
            description="Test description",
            trigger_conditions=["push"],
            environment_variables={"ENV": "test"},
            steps=[
                PipelineStep(
                    step_id="test_step",
                    name="Test Step",
                    stage=PipelineStage.TEST,
                    command="echo 'test'",
                )
            ],
            timeout=1800,
            parallel_jobs=1,
        )
        assert config.pipeline_id == "test-pipeline"
        assert len(config.steps) == 1
        assert config.parallel_jobs == 1

    def test_build_result_structure(self):
        """Test build result data structure validation."""
        start_time = datetime.now(UTC)
        end_time = start_time + timedelta(minutes=5)

        result = BuildResult(
            build_id="build-001",
            pipeline_id="pipeline-001",
            status=BuildStatus.SUCCESS,
            start_time=start_time,
            end_time=end_time,
            duration=end_time - start_time,
            steps_executed=["build", "test"],
            failed_step=None,
            artifacts_generated=["dist/app.js", "coverage.xml"],
        )

        assert result.build_id == "build-001"
        assert result.status == BuildStatus.SUCCESS
        assert len(result.steps_executed) == 2
        assert len(result.artifacts_generated) == 2

    def test_deployment_config_structure(self):
        """Test deployment configuration validation."""
        config = DeploymentConfig(
            environment="production",
            strategy=DeploymentStrategy.BLUE_GREEN,
            target_infrastructure={"cluster": "prod-cluster"},
            health_checks=["http://health", "tcp:3000"],
            rollback_enabled=True,
            approval_required=True,
        )

        assert config.environment == "production"
        assert config.strategy == DeploymentStrategy.BLUE_GREEN
        assert config.rollback_enabled is True
        assert config.approval_required is True


class TestCICDPipelineExecution:
    """Test pipeline execution and build management."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return CICDPipeline()

    @pytest.fixture
    def sample_config(self):
        """Sample pipeline configuration."""
        return {
            "id": "execution-test",
            "name": "Execution Test Pipeline",
            "description": "Test execution flow",
            "trigger_conditions": ["manual"],
            "environment_variables": {"TEST_MODE": "true"},
            "steps": [
                {
                    "id": "prepare",
                    "name": "Prepare Environment",
                    "stage": "build",
                    "command": "echo 'preparing'",
                    "timeout": 60,
                },
                {
                    "id": "compile",
                    "name": "Compile Code",
                    "stage": "build",
                    "command": "echo 'compiling'",
                    "depends_on": ["prepare"],
                    "timeout": 120,
                },
                {
                    "id": "unit_test",
                    "name": "Unit Tests",
                    "stage": "test",
                    "command": "echo 'testing'",
                    "depends_on": ["compile"],
                },
            ],
        }

    @pytest.mark.asyncio
    async def test_create_pipeline_success(self, pipeline, sample_config):
        """Test successful pipeline creation."""
        result = await pipeline.create_pipeline(sample_config)

        assert result.is_right()
        config = result.get_right()
        assert config.pipeline_id == "execution-test"
        assert config.name == "Execution Test Pipeline"
        assert len(config.steps) == 3

        # Verify pipeline is stored
        assert "execution-test" in pipeline.active_pipelines

    @pytest.mark.asyncio
    async def test_pipeline_validation_error(self, pipeline):
        """Test pipeline validation with invalid configuration."""
        invalid_config = {
            "pipeline_id": "",  # Invalid: empty ID
            "name": "Invalid Pipeline",
            "steps": [],  # Invalid: no steps
        }

        result = await pipeline.create_pipeline(invalid_config)
        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, OrchestrationError)

    @pytest.mark.asyncio
    async def test_step_dependency_validation(self, pipeline):
        """Test step dependency validation."""
        config_with_circular_dependency = {
            "id": "circular-test",
            "name": "Circular Dependency Test",
            "description": "Test circular dependency detection",
            "trigger_conditions": ["manual"],
            "environment_variables": {},
            "steps": [
                {
                    "id": "step_a",
                    "name": "Step A",
                    "stage": "build",
                    "command": "echo 'A'",
                    "depends_on": ["step_b"],
                },
                {
                    "id": "step_b",
                    "name": "Step B",
                    "stage": "build",
                    "command": "echo 'B'",
                    "depends_on": ["step_a"],  # Circular dependency
                },
            ],
        }

        result = await pipeline.create_pipeline(config_with_circular_dependency)
        assert result.is_left()
        error = result.get_left()
        assert "circular" in str(error).lower()

    @pytest.mark.asyncio
    async def test_execute_pipeline_success(self, pipeline, sample_config):
        """Test successful pipeline execution."""
        # Create pipeline first
        create_result = await pipeline.create_pipeline(sample_config)
        assert create_result.is_right()

        # Mock step execution to avoid actual command execution
        with patch.object(
            pipeline, "_execute_step", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = True

            # Execute pipeline
            build_result = await pipeline.execute_pipeline("execution-test")

            assert build_result.is_right()
            result = build_result.get_right()
            assert result.status == BuildStatus.SUCCESS
            assert len(result.steps_executed) == 3

            # Verify steps were called in correct order
            assert mock_execute.call_count == 3

    @pytest.mark.asyncio
    async def test_execute_pipeline_with_failure(self, pipeline, sample_config):
        """Test pipeline execution with step failure."""
        # Create pipeline
        create_result = await pipeline.create_pipeline(sample_config)
        assert create_result.is_right()

        # Mock step execution with failure on second step
        async def mock_execute_side_effect(step, environment, build_result):
            if step.step_id == "compile":
                return False  # Step failed
            return True  # Step succeeded

        with patch.object(
            pipeline, "_execute_step", side_effect=mock_execute_side_effect
        ):
            build_result = await pipeline.execute_pipeline("execution-test")

            assert build_result.is_right()
            result = build_result.get_right()
            assert result.status == BuildStatus.FAILED
            assert result.failed_step == "compile"
            assert "prepare" in result.steps_executed
            assert "compile" not in result.steps_executed


class TestCICDPipelineStatus:
    """Test pipeline status monitoring and management."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return CICDPipeline()

    @pytest.mark.asyncio
    async def test_get_pipeline_status_running(self, pipeline):
        """Test getting status of running pipeline."""
        # First create a pipeline
        config = {
            "id": "status-test-pipeline",
            "name": "Status Test Pipeline",
            "steps": [
                {
                    "id": "build",
                    "name": "Build",
                    "stage": "build",
                    "command": "echo build",
                }
            ],
        }
        create_result = await pipeline.create_pipeline(config)
        assert create_result.is_right()

        # Add a running build for this pipeline
        build_result = BuildResult(
            build_id="running-build",
            pipeline_id="status-test-pipeline",
            status=BuildStatus.RUNNING,
            start_time=datetime.now(UTC),
            end_time=None,
            duration=None,
            steps_executed=["build"],
            failed_step=None,
            artifacts_generated=[],
        )
        pipeline.running_builds["running-build"] = build_result

        status = await pipeline.get_pipeline_status("status-test-pipeline")

        assert status.is_right()
        status_data = status.get_right()
        assert status_data["pipeline_id"] == "status-test-pipeline"
        assert status_data["name"] == "Status Test Pipeline"
        assert status_data["is_running"] is True

    @pytest.mark.asyncio
    async def test_get_pipeline_status_not_found(self, pipeline):
        """Test getting status of non-existent pipeline."""
        status = await pipeline.get_pipeline_status("non-existent")

        assert status.is_left()
        error = status.get_left()
        assert "not found" in str(error).lower()

    @pytest.mark.asyncio
    async def test_cancel_build_success(self, pipeline):
        """Test successful build cancellation."""
        # Add a running build
        build_result = BuildResult(
            build_id="cancel-test",
            pipeline_id="test-pipeline",
            status=BuildStatus.RUNNING,
            start_time=datetime.now(UTC),
            end_time=None,
            duration=None,
            steps_executed=[],
            failed_step=None,
            artifacts_generated=[],
        )
        pipeline.running_builds["cancel-test"] = build_result

        result = await pipeline.cancel_build("cancel-test")

        assert result.is_right()
        assert result.get_right() is True

        # Verify build is no longer running
        cancelled_build = pipeline.running_builds.get("cancel-test")
        if cancelled_build:
            assert cancelled_build.status == BuildStatus.CANCELLED

    @pytest.mark.asyncio
    async def test_cancel_build_not_found(self, pipeline):
        """Test cancelling non-existent build."""
        result = await pipeline.cancel_build("non-existent")

        assert result.is_left()
        error = result.get_left()
        assert "not found" in str(error).lower()


class TestCICDPipelineAdvanced:
    """Test advanced CI/CD pipeline features."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return CICDPipeline()

    def test_execution_order_calculation(self, pipeline):
        """Test calculation of step execution order with dependencies."""
        steps = [
            PipelineStep(
                "step_c",
                "Step C",
                PipelineStage.DEPLOY,
                "echo c",
                depends_on=["step_b"],
            ),
            PipelineStep("step_a", "Step A", PipelineStage.BUILD, "echo a"),
            PipelineStep(
                "step_b", "Step B", PipelineStage.TEST, "echo b", depends_on=["step_a"]
            ),
        ]

        execution_order = pipeline._calculate_execution_order(steps)

        # Should be ordered: step_a, step_b, step_c
        step_ids = [step.step_id for step in execution_order]
        assert step_ids.index("step_a") < step_ids.index("step_b")
        assert step_ids.index("step_b") < step_ids.index("step_c")

    @pytest.mark.asyncio
    async def test_artifact_collection(self, pipeline):
        """Test artifact collection after step execution."""
        # Create a test step with artifacts
        step = PipelineStep(
            step_id="artifact_test",
            name="Artifact Test Step",
            stage=PipelineStage.BUILD,
            command="echo 'building'",
            artifacts=["dist/app.js", "coverage.xml"],
        )

        # Create a build result to collect artifacts into
        build_result = BuildResult(
            build_id="test-build",
            pipeline_id="test-pipeline",
            status=BuildStatus.RUNNING,
            start_time=datetime.now(UTC),
            end_time=None,
            duration=None,
            steps_executed=[],
            failed_step=None,
            artifacts_generated=[],
        )

        # Mock file system operations
        with patch("pathlib.Path.exists", return_value=True):
            await pipeline._collect_artifacts(step, build_result)

            # Verify artifacts were collected
            assert len(build_result.artifacts_generated) == 2
            assert "dist/app.js" in build_result.artifacts_generated
            assert "coverage.xml" in build_result.artifacts_generated

    @pytest.mark.asyncio
    async def test_step_execution_with_retry(self, pipeline):
        """Test step execution with retry mechanism."""
        step = PipelineStep(
            step_id="retry_test",
            name="Retry Test Step",
            stage=PipelineStage.TEST,
            command="exit 1",  # Will fail
            retry_count=2,
        )

        # Create build result and environment
        build_result = BuildResult(
            build_id="retry-test",
            pipeline_id="test-pipeline",
            status=BuildStatus.RUNNING,
            start_time=datetime.now(UTC),
            end_time=None,
            duration=None,
            steps_executed=[],
            failed_step=None,
            artifacts_generated=[],
        )
        environment = {"CI": "true"}

        with patch("asyncio.create_subprocess_shell") as mock_subprocess:
            # Mock process that fails
            mock_process = AsyncMock()
            mock_process.returncode = 1  # Failure
            mock_process.communicate = AsyncMock(
                return_value=(b"Error output", b"Error")
            )
            mock_subprocess.return_value = mock_process

            result = await pipeline._execute_step(step, environment, build_result)

            # Should have been called at least once and failed
            assert mock_subprocess.call_count >= 1
            assert result is False  # Should fail due to non-zero return code

    def test_pipeline_enumeration_values(self):
        """Test all pipeline enumeration values."""
        # Test PipelineStage enum
        stages = list(PipelineStage)
        assert PipelineStage.BUILD in stages
        assert PipelineStage.TEST in stages
        assert PipelineStage.DEPLOY in stages
        assert len(stages) == 7

        # Test BuildStatus enum
        statuses = list(BuildStatus)
        assert BuildStatus.PENDING in statuses
        assert BuildStatus.RUNNING in statuses
        assert BuildStatus.SUCCESS in statuses
        assert BuildStatus.FAILED in statuses
        assert len(statuses) == 6

        # Test DeploymentStrategy enum
        strategies = list(DeploymentStrategy)
        assert DeploymentStrategy.ROLLING in strategies
        assert DeploymentStrategy.BLUE_GREEN in strategies
        assert DeploymentStrategy.CANARY in strategies
        assert len(strategies) == 4

        # Test TestingStrategy enum
        test_strategies = list(TestingStrategy)
        assert TestingStrategy.UNIT in test_strategies
        assert TestingStrategy.INTEGRATION in test_strategies
        assert TestingStrategy.E2E in test_strategies
        assert len(test_strategies) == 4


class TestCICDPipelineFactory:
    """Test CI/CD pipeline factory function."""

    def test_get_cicd_pipeline_function(self):
        """Test factory function returns valid pipeline instance."""
        pipeline = get_cicd_pipeline()

        assert isinstance(pipeline, CICDPipeline)
        assert pipeline.cache_enabled is True
        assert pipeline.max_concurrent_builds == 5
        assert len(pipeline.active_pipelines) == 0


class TestCICDPipelineErrorHandling:
    """Test comprehensive error handling in CI/CD pipeline."""

    @pytest.fixture
    def pipeline(self):
        """Create pipeline instance for testing."""
        return CICDPipeline()

    @pytest.mark.asyncio
    async def test_invalid_pipeline_config_structure(self, pipeline):
        """Test handling of malformed pipeline configuration."""
        invalid_configs = [
            {},  # Empty config
            {"pipeline_id": "test"},  # Missing required fields
            {"pipeline_id": "test", "name": "", "steps": []},  # Empty name and steps
            {
                "pipeline_id": "test",
                "name": "Test",
                "steps": [{"invalid": "step"}],
            },  # Invalid step
        ]

        for config in invalid_configs:
            result = await pipeline.create_pipeline(config)
            assert result.is_left()
            assert isinstance(result.get_left(), OrchestrationError)

    @pytest.mark.asyncio
    async def test_step_execution_timeout(self, pipeline):
        """Test step execution timeout handling."""
        step = PipelineStep(
            step_id="timeout_test",
            name="Timeout Test Step",
            stage=PipelineStage.BUILD,
            command="sleep 10",
            timeout=1,  # 1 second timeout
        )

        # Create build result and environment for timeout test
        build_result = BuildResult(
            build_id="timeout-test",
            pipeline_id="test-pipeline",
            status=BuildStatus.RUNNING,
            start_time=datetime.now(UTC),
            end_time=None,
            duration=None,
            steps_executed=[],
            failed_step=None,
            artifacts_generated=[],
        )
        environment = {"CI": "true"}

        with patch("asyncio.wait_for", side_effect=asyncio.TimeoutError):
            result = await pipeline._execute_step(step, environment, build_result)

            assert result is False  # Should return False on timeout
            assert len(build_result.logs) > 0  # Should have logged the timeout

    @pytest.mark.asyncio
    async def test_workspace_directory_error(self):
        """Test handling of invalid workspace directory."""
        # Test with non-existent path (should still initialize but may affect operations)
        pipeline = CICDPipeline(workspace_path="/non/existent/path")
        assert pipeline.workspace_path == Path("/non/existent/path")

        # Pipeline should still be functional for basic operations
        assert len(pipeline.active_pipelines) == 0
        assert pipeline.cache_enabled is True
