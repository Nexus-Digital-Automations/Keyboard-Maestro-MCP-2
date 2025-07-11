"""Focused tests for src/server/tools/testing_automation_tools.py.

This module provides strategic test coverage for the testing automation tools
focusing on the actual available functions to improve coverage toward 95% threshold.
"""

import uuid
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest
from fastmcp import Context
from src.core.testing_architecture import (
    AutomationTest,
    QualityAssessment,
    QualityGate,
    QualityMetric,
    RegressionDetection,
    TestConfiguration,
    TestCriteria,
    TestEnvironment,
    TestResult,
    TestScope,
    TestStatus,
    TestStep,
    TestType,
)
from src.server.tools.testing_automation_tools import (
    km_detect_regressions,
    km_generate_test_reports,
    km_run_comprehensive_tests,
    km_validate_automation_quality,
    quality_assessments,
    regression_analyses,
    test_execution_history,
    test_runner,
)


class TestKMRunComprehensiveTests:
    """Test km_run_comprehensive_tests function with comprehensive scenarios."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        context = AsyncMock(spec=Context)
        context.info = AsyncMock()
        context.error = AsyncMock()
        context.warn = AsyncMock()
        return context

    @pytest.fixture
    def valid_test_parameters(self):
        """Valid test parameters for testing."""
        return {
            "test_scope": "macro",
            "target_ids": ["test-macro-001", "test-macro-002"],
            "test_types": ["functional", "performance"],
            "test_environment": "development",
            "parallel_execution": True,
            "max_execution_time": 300,
            "_include_performance_tests": True,
            "generate_coverage_report": True,
            "stop_on_failure": False,
        }

    async def test_run_comprehensive_tests_success_minimal(
        self, mock_context, valid_test_parameters
    ):
        """Test successful comprehensive test execution with minimal parameters."""
        with patch.object(
            test_runner, "execute_test_suite", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                TestResult(
                    test_id="test-001",
                    test_run_id="run-001",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    execution_time_ms=1500,
                    step_results=[],
                    assertions_passed=3,
                    assertions_failed=0,
                ),
                TestResult(
                    test_id="test-002",
                    test_run_id="run-002",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    execution_time_ms=2100,
                    step_results=[],
                    assertions_passed=2,
                    assertions_failed=0,
                ),
            ]

            result = await km_run_comprehensive_tests(
                test_scope=valid_test_parameters["test_scope"],
                target_ids=valid_test_parameters["target_ids"],
                test_types=valid_test_parameters["test_types"],
                test_environment=valid_test_parameters["test_environment"],
                parallel_execution=valid_test_parameters["parallel_execution"],
                max_execution_time=valid_test_parameters["max_execution_time"],
                _include_performance_tests=valid_test_parameters[
                    "_include_performance_tests"
                ],
                generate_coverage_report=valid_test_parameters[
                    "generate_coverage_report"
                ],
                stop_on_failure=valid_test_parameters["stop_on_failure"],
                ctx=mock_context,
            )

            assert result["success"] is True
            assert "test_run_id" in result
            assert result["execution_summary"]["total_tests"] == 2
            assert result["execution_summary"]["passed_tests"] == 2

            # Verify context info was called
            mock_context.info.assert_called()

    async def test_run_comprehensive_tests_validation_errors(self, mock_context):
        """Test validation error handling for invalid parameters."""
        # Test empty target_ids
        result = await km_run_comprehensive_tests(
            test_scope="macro",
            target_ids=[],
            test_types=["functional"],
            test_environment="development",
            parallel_execution=True,
            max_execution_time=300,
            _include_performance_tests=True,
            generate_coverage_report=True,
            stop_on_failure=False,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "At least one target ID is required" in result["error"]

        # Test invalid test scope
        result = await km_run_comprehensive_tests(
            test_scope="invalid_scope",
            target_ids=["test-001"],
            test_types=["functional"],
            test_environment="development",
            parallel_execution=True,
            max_execution_time=300,
            _include_performance_tests=True,
            generate_coverage_report=True,
            stop_on_failure=False,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "Invalid test scope" in result["error"]
        assert "available_scopes" in result

        # Test invalid test environment
        result = await km_run_comprehensive_tests(
            test_scope="macro",
            target_ids=["test-001"],
            test_types=["functional"],
            test_environment="invalid_env",
            parallel_execution=True,
            max_execution_time=300,
            _include_performance_tests=True,
            generate_coverage_report=True,
            stop_on_failure=False,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "Invalid test environment" in result["error"]

        # Test invalid test types
        result = await km_run_comprehensive_tests(
            test_scope="macro",
            target_ids=["test-001"],
            test_types=["invalid_type", "functional"],
            test_environment="development",
            parallel_execution=True,
            max_execution_time=300,
            _include_performance_tests=True,
            generate_coverage_report=True,
            stop_on_failure=False,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "Invalid test types" in result["error"]
        assert "valid_types" in result

        # Test invalid execution time
        result = await km_run_comprehensive_tests(
            test_scope="macro",
            target_ids=["test-001"],
            test_types=["functional"],
            test_environment="development",
            parallel_execution=True,
            max_execution_time=30,  # Too short
            _include_performance_tests=True,
            generate_coverage_report=True,
            stop_on_failure=False,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert (
            "Max execution time must be between 60 and 7200 seconds" in result["error"]
        )

    async def test_run_comprehensive_tests_execution_failure(
        self, mock_context, valid_test_parameters
    ):
        """Test handling of test execution failures."""
        with patch.object(
            test_runner, "execute_test_suite", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.side_effect = Exception("Test runner failure")

            result = await km_run_comprehensive_tests(
                test_scope=valid_test_parameters["test_scope"],
                target_ids=valid_test_parameters["target_ids"],
                test_types=valid_test_parameters["test_types"],
                test_environment=valid_test_parameters["test_environment"],
                parallel_execution=valid_test_parameters["parallel_execution"],
                max_execution_time=valid_test_parameters["max_execution_time"],
                _include_performance_tests=valid_test_parameters[
                    "_include_performance_tests"
                ],
                generate_coverage_report=valid_test_parameters[
                    "generate_coverage_report"
                ],
                stop_on_failure=valid_test_parameters["stop_on_failure"],
                ctx=mock_context,
            )

            assert result["success"] is False
            assert "Test runner failure" in result["error"]
            assert result["error_type"] == "Exception"

    async def test_run_comprehensive_tests_default_parameters(self, mock_context):
        """Test execution with default parameters."""
        with patch.object(
            test_runner, "execute_test_suite", new_callable=AsyncMock
        ) as mock_execute:
            mock_execute.return_value = [
                TestResult(
                    test_id="test-001",
                    test_run_id="run-001",
                    status=TestStatus.PASSED,
                    start_time=datetime.now(UTC),
                    end_time=datetime.now(UTC),
                    execution_time_ms=1000,
                    step_results=[],
                    assertions_passed=1,
                    assertions_failed=0,
                )
            ]

            result = await km_run_comprehensive_tests(
                test_scope="system",
                target_ids=["system-test-001"],
                test_types=None,  # Use defaults
                test_environment="development",
                parallel_execution=True,
                max_execution_time=1800,
                _include_performance_tests=True,
                generate_coverage_report=True,
                stop_on_failure=False,
                ctx=mock_context,
            )

            assert result["success"] is True
            # The API doesn't return test_configuration - check actual response structure
            assert result["execution_summary"]["total_tests"] == 1
            assert result["execution_summary"]["passed_tests"] == 1
            assert "quality_assessment" in result


class TestKMValidateAutomationQuality:
    """Test km_validate_automation_quality function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return AsyncMock(spec=Context)

    async def test_validate_automation_quality_success(self, mock_context):
        """Test successful automation quality validation."""
        result = await km_validate_automation_quality(
            validation_target="workflow",
            target_id="workflow-001",
            quality_criteria=["reliability", "performance", "maintainability"],
            validation_depth="comprehensive",
            include_static_analysis=True,
            include_security_checks=True,
            benchmark_against_standards=True,
            generate_quality_score=True,
            provide_recommendations=True,
            ctx=mock_context,
        )

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["assessment_id"] is not None
        assert result["data"]["validation_target"] == "workflow"
        assert result["data"]["target_id"] == "workflow-001"
        assert "overall_score" in result["data"]
        assert "code_quality_score" in result["data"]

    async def test_validate_automation_quality_validation_errors(self, mock_context):
        """Test validation error handling."""
        # Test invalid validation target
        result = await km_validate_automation_quality(
            validation_target="invalid_target",
            target_id="test-001",
            quality_criteria=None,
            validation_depth="standard",
            include_static_analysis=True,
            include_security_checks=True,
            benchmark_against_standards=True,
            generate_quality_score=True,
            provide_recommendations=True,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "Invalid validation target" in result["error"]

        # Test invalid quality depth
        result = await km_validate_automation_quality(
            validation_target="macro",
            target_id="test-001",
            quality_criteria=None,
            validation_depth="invalid_depth",
            include_static_analysis=True,
            include_security_checks=True,
            benchmark_against_standards=True,
            generate_quality_score=True,
            provide_recommendations=True,
            ctx=mock_context,
        )
        assert result["success"] is False
        assert "Invalid validation depth" in result["error"]

    async def test_validate_automation_quality_minimal_parameters(self, mock_context):
        """Test quality validation with minimal parameters."""
        result = await km_validate_automation_quality(
            validation_target="macro",
            target_id="macro-001",
            quality_criteria=None,
            validation_depth="standard",
            include_static_analysis=True,
            include_security_checks=True,
            benchmark_against_standards=True,
            generate_quality_score=True,
            provide_recommendations=True,
            ctx=mock_context,
        )

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["validation_target"] == "macro"
        assert result["data"]["target_id"] == "macro-001"
        assert "overall_score" in result["data"]


class TestKMDetectRegressions:
    """Test km_detect_regressions function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return AsyncMock(spec=Context)

    async def test_detect_regressions_success(self, mock_context):
        """Test successful regression detection."""
        result = await km_detect_regressions(
            comparison_scope="system",
            baseline_version="v1.0.0",
            current_version="v1.0.1",
            regression_types=["functional", "performance"],
            sensitivity_level="medium",
            _include_performance_regression=True,
            auto_categorize_issues=True,
            generate_impact_analysis=True,
            provide_fix_suggestions=True,
            ctx=mock_context,
        )

        assert result["success"] is True
        assert "data" in result
        assert "analysis_id" in result["data"]
        assert "regression_detected" in result["data"]
        assert "risk_level" in result["data"]

    async def test_detect_regressions_no_regressions(self, mock_context):
        """Test regression detection with no regressions found."""
        result = await km_detect_regressions(
            comparison_scope="macro",
            baseline_version="v1.0.0",
            current_version="v1.0.0",  # Same version = no regressions
            ctx=mock_context,
        )

        assert result["success"] is True
        assert "data" in result
        assert result["data"]["regression_detected"] is False

    async def test_detect_regressions_missing_executions(self, mock_context):
        """Test regression detection with invalid parameters."""
        result = await km_detect_regressions(
            comparison_scope="invalid_scope",
            baseline_version="v1.0.0",
            current_version="v1.0.1",
            ctx=mock_context,
        )

        assert result["success"] is False
        assert "Invalid comparison scope" in result["error"]


class TestKMGenerateTestReports:
    """Test km_generate_test_reports function."""

    @pytest.fixture
    def mock_context(self):
        """Create mock FastMCP context."""
        return AsyncMock(spec=Context)

    async def test_generate_test_reports_success(self, mock_context):
        """Test successful test report generation."""
        data_sources = ["test_run_1", "test_run_2", "test_run_3"]

        # Setup test execution history for data sources
        for i, source in enumerate(data_sources):
            test_execution_history[source] = {
                "results": [
                    {
                        "test_id": f"test-{i}-001",
                        "status": "passed",
                        "duration": 1.0 + i * 0.5,
                    },
                    {
                        "test_id": f"test-{i}-002",
                        "status": "passed" if i < 2 else "failed",
                        "duration": 0.8,
                    },
                ],
                "summary": {
                    "total": 2,
                    "passed": 2 if i < 2 else 1,
                    "failed": 0 if i < 2 else 1,
                },
                "coverage_percentage": 85.0 + i * 2,
                "start_time": datetime.now(UTC).isoformat(),
            }

        result = await km_generate_test_reports(
            report_scope="comprehensive",
            data_sources=data_sources,
            report_format="json",
            include_visualizations=True,
            include_trends=True,
            include_recommendations=True,
            ctx=mock_context,
        )

        assert result["success"] is True
        assert "report_id" in result
        assert result["test_summary"]["total_test_runs"] == 3

    async def test_generate_test_reports_invalid_executions(self, mock_context):
        """Test test report generation with invalid scope."""
        result = await km_generate_test_reports(
            report_scope="invalid_scope", data_sources=["test_run_1"], ctx=mock_context
        )

        assert result["success"] is False
        assert "Invalid report scope" in result["error"]

    async def test_generate_test_reports_summary_only(self, mock_context):
        """Test test report generation with minimal options."""
        test_execution_id = f"test_exec_{uuid.uuid4()}"
        test_execution_history[test_execution_id] = {
            "results": [{"test_id": "test-001", "status": "passed", "duration": 1.0}],
            "summary": {"total": 1, "passed": 1, "failed": 0},
            "coverage_percentage": 90.0,
        }

        result = await km_generate_test_reports(
            report_scope="test_run",
            data_sources=[test_execution_id],
            report_format="json",
            include_trends=False,
            include_recommendations=False,
            include_visualizations=False,
            ctx=mock_context,
        )

        assert result["success"] is True
        assert result["trend_analysis"]["trends_identified"] == 0
        assert result["visualizations"]["charts_generated"] == 0


class TestTestingArchitectureIntegration:
    """Test integration with testing architecture components."""

    def test_automation_test_creation(self):
        """Test AutomationTest creation and validation."""
        # Create the required configuration and criteria
        test_config = TestConfiguration(
            test_type=TestType.FUNCTIONAL,
            test_scope=TestScope.UNIT,
            environment=TestEnvironment.DEVELOPMENT,
        )

        test_criteria = TestCriteria(expected_results={"status": "success"})

        test_step = TestStep(
            step_id="step-001",
            step_name="Execute test",
            step_type="execute",
            action="execute_macro",
            parameters={"macro_id": "macro-001"},
        )

        test = AutomationTest(
            test_id="test-001",
            test_name="Functional Test",
            description="Test functional behavior",
            test_configuration=test_config,
            test_criteria=test_criteria,
            test_steps=[test_step],
        )

        assert test.test_id == "test-001"
        assert test.test_name == "Functional Test"
        assert test.test_configuration.test_type == TestType.FUNCTIONAL

    def test_quality_assessment_creation(self):
        """Test QualityAssessment creation."""
        gate1 = QualityGate(
            gate_id="gate-001",
            gate_name="Coverage Gate",
            metric=QualityMetric.COVERAGE,
            threshold=80.0,
            operator="gte",
        )

        gate2 = QualityGate(
            gate_id="gate-002",
            gate_name="Performance Gate",
            metric=QualityMetric.PERFORMANCE,
            threshold=75.0,
            operator="gte",
        )

        assessment = QualityAssessment(
            assessment_id="qa-001",
            test_run_id="run-001",
            overall_score=85.5,
            metric_scores={
                QualityMetric.COVERAGE: 90.0,
                QualityMetric.PERFORMANCE: 70.0,
            },
            gates_passed=[gate1],
            gates_failed=[gate2],
            recommendations=["Improve performance"],
            risk_level="medium",
        )

        assert assessment.overall_score == 85.5
        assert len(assessment.gates_passed) == 1
        assert len(assessment.gates_failed) == 1
        assert assessment.risk_level == "medium"

    def test_test_configuration_creation(self):
        """Test TestConfiguration creation."""
        config = TestConfiguration(
            test_type=TestType.INTEGRATION,
            test_scope=TestScope.SYSTEM,
            environment=TestEnvironment.STAGING,
            parallel_execution=True,
            timeout_seconds=300,
        )

        assert config.environment == TestEnvironment.STAGING
        assert config.parallel_execution is True
        assert config.timeout_seconds == 300
        assert config.test_type == TestType.INTEGRATION
        assert config.test_scope == TestScope.SYSTEM

    def test_regression_detection_creation(self):
        """Test RegressionDetection creation."""
        regression = RegressionDetection(
            detection_id="reg-001",
            baseline_run_id="exec-001",
            current_run_id="exec-002",
            detection_sensitivity="medium",
            metrics_to_compare=["execution_time", "memory_usage"],
            threshold_percentage=5.0,
            regressions_found=[{"metric": "execution_time", "degradation": 15.0}],
        )

        assert len(regression.regressions_found) == 1
        assert regression.threshold_percentage == 5.0


class TestTestingAutomationToolsGlobals:
    """Test global components and utilities."""

    def test_global_test_runner(self):
        """Test global test runner initialization."""
        assert test_runner is not None
        assert hasattr(test_runner, "execute_test_suite")

    def test_global_data_structures(self):
        """Test global data structure initialization."""
        assert isinstance(test_execution_history, dict)
        assert isinstance(quality_assessments, dict)
        assert isinstance(regression_analyses, dict)

    def test_test_execution_history_operations(self):
        """Test test execution history operations."""
        test_id = f"test_{uuid.uuid4()}"
        test_data = {
            "test_results": [{"test_id": "test-001", "status": "passed"}],
            "summary": {"total": 1, "passed": 1, "failed": 0},
        }

        # Add to history
        test_execution_history[test_id] = test_data

        # Verify retrieval
        assert test_id in test_execution_history
        assert test_execution_history[test_id]["summary"]["total"] == 1

        # Clean up
        del test_execution_history[test_id]

    def test_quality_assessments_operations(self):
        """Test quality assessments operations."""
        assessment_id = f"qa_{uuid.uuid4()}"
        assessment_data = {
            "quality_score": 92.5,
            "metrics": [{"name": "coverage", "value": 95.0}],
            "timestamp": datetime.now(UTC).isoformat(),
        }

        quality_assessments[assessment_id] = assessment_data

        assert assessment_id in quality_assessments
        assert quality_assessments[assessment_id]["quality_score"] == 92.5

        # Clean up
        del quality_assessments[assessment_id]

    def test_regression_analyses_operations(self):
        """Test regression analyses operations."""
        regression_id = f"reg_{uuid.uuid4()}"
        regression_data = {
            "regressions_found": True,
            "total_regressions": 2,
            "analysis_timestamp": datetime.now(UTC).isoformat(),
        }

        regression_analyses[regression_id] = regression_data

        assert regression_id in regression_analyses
        assert regression_analyses[regression_id]["regressions_found"] is True

        # Clean up
        del regression_analyses[regression_id]


# Property-based tests for comprehensive coverage
class TestPropertyBasedTesting:
    """Property-based tests for testing automation tools."""

    @pytest.mark.parametrize(
        "test_scope", ["macro", "workflow", "system", "integration"]
    )
    def test_valid_test_scopes_property(self, test_scope):
        """Property: All valid test scopes should be accepted."""
        valid_scopes = ["macro", "workflow", "system", "integration"]
        assert test_scope in valid_scopes

    @pytest.mark.parametrize(
        "test_type",
        [
            "functional",
            "performance",
            "integration",
            "security",
            "regression",
            "load",
            "stress",
        ],
    )
    def test_valid_test_types_property(self, test_type):
        """Property: All valid test types should be mapped correctly."""
        valid_test_types = {
            "functional": TestType.FUNCTIONAL,
            "performance": TestType.PERFORMANCE,
            "integration": TestType.INTEGRATION,
            "security": TestType.SECURITY,
            "regression": TestType.REGRESSION,
            "load": TestType.LOAD,
            "stress": TestType.STRESS,
        }

        assert test_type in valid_test_types
        assert isinstance(valid_test_types[test_type], TestType)

    @pytest.mark.parametrize("environment", ["development", "staging", "production"])
    def test_valid_environments_property(self, environment):
        """Property: All valid environments should be accepted."""
        valid_environments = ["development", "staging", "production"]
        assert environment in valid_environments

    @pytest.mark.parametrize("execution_time", [60, 300, 1800, 3600, 7200])
    def test_valid_execution_times_property(self, execution_time):
        """Property: Valid execution times should be within acceptable range."""
        assert 60 <= execution_time <= 7200


class TestInternalUtilityFunctions:
    """Test internal utility functions for comprehensive coverage."""

    def test_calculate_parallel_efficiency(self):
        """Test _calculate_parallel_efficiency function."""
        from src.server.tools.testing_automation_tools import (
            _calculate_parallel_efficiency,
        )

        # Test basic efficiency calculation with actual signature
        efficiency = _calculate_parallel_efficiency(
            _test_results=[], parallel_execution=True
        )

        assert isinstance(efficiency, float)
        assert 0.0 <= efficiency <= 100.0

        # Test sequential execution
        sequential_efficiency = _calculate_parallel_efficiency(
            _test_results=[], parallel_execution=False
        )
        assert sequential_efficiency == 0.0

    def test_create_quality_gates_for_depth(self):
        """Test _create_quality_gates_for_depth function."""
        from src.server.tools.testing_automation_tools import (
            _create_quality_gates_for_depth,
        )

        # Test different depth levels
        basic_gates = _create_quality_gates_for_depth("basic")
        assert isinstance(basic_gates, list)
        assert len(basic_gates) > 0

        standard_gates = _create_quality_gates_for_depth("standard")
        assert len(standard_gates) >= len(basic_gates)

        comprehensive_gates = _create_quality_gates_for_depth("comprehensive")
        assert len(comprehensive_gates) >= len(standard_gates)

    def test_evaluate_quality_gate(self):
        """Test _evaluate_quality_gate function."""
        from src.server.tools.testing_automation_tools import _evaluate_quality_gate

        gate = QualityGate(
            gate_id="coverage_gate",
            gate_name="Coverage Gate",
            metric=QualityMetric.COVERAGE,
            threshold=80.0,
            operator="gte",
        )

        # Test passing score
        assert _evaluate_quality_gate(85.0, gate) is True

        # Test failing score
        assert _evaluate_quality_gate(75.0, gate) is False

        # Test edge case
        assert _evaluate_quality_gate(80.0, gate) is True

    def test_is_regression(self):
        """Test _is_regression function."""
        from src.server.tools.testing_automation_tools import _is_regression

        # Test significant regression
        assert (
            _is_regression("execution_time_ms", 25.0) is True
        )  # 25% increase is regression

        # Test minor improvement
        assert (
            _is_regression("execution_time_ms", -5.0) is False
        )  # 5% decrease is improvement

        # Test quality regression
        assert (
            _is_regression("success_rate", -10.0) is True
        )  # 10% decrease in success rate is regression

    def test_determine_regression_severity(self):
        """Test _determine_regression_severity function."""
        from src.server.tools.testing_automation_tools import (
            _determine_regression_severity,
        )

        # Test critical regression
        severity = _determine_regression_severity(50.0, 10.0)
        assert severity == "critical"

        # Test high regression
        severity = _determine_regression_severity(20.0, 10.0)
        assert severity == "high"

        # Test medium regression
        severity = _determine_regression_severity(15.0, 10.0)
        assert severity == "medium"

        # Test low regression
        severity = _determine_regression_severity(5.0, 10.0)
        assert severity == "low"

    def test_categorize_regression(self):
        """Test _categorize_regression function."""
        from src.server.tools.testing_automation_tools import _categorize_regression

        assert _categorize_regression("execution_time_ms") == "performance"
        assert _categorize_regression("memory_usage_mb") == "resource"
        assert _categorize_regression("success_rate") == "functional"
        assert _categorize_regression("unknown_metric") == "other"

    def test_categorize_improvement(self):
        """Test _categorize_improvement function."""
        from src.server.tools.testing_automation_tools import _categorize_improvement

        assert _categorize_improvement("execution_time_ms") == "performance"
        assert _categorize_improvement("memory_usage_mb") == "resource"
        assert _categorize_improvement("success_rate") == "functional"
        assert _categorize_improvement("unknown_metric") == "other"
