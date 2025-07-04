"""
Testing Architecture - TASK_58 Phase 1 Implementation

Comprehensive testing type definitions, test orchestration patterns, and quality assurance frameworks.
Extends the existing macro testing framework with advanced automation testing capabilities.

Architecture: Testing Types + Test Orchestration + Quality Metrics + Security Validation
Performance: <100ms test setup, <5s complex test suites, <1s quality analysis
Security: Safe test execution, isolated environments, comprehensive validation
"""

from __future__ import annotations
from typing import Dict, List, Optional, Any, Set, Union, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC
from abc import ABC, abstractmethod
from enum import Enum
import uuid
import asyncio

from src.core.types import MacroId, create_macro_id
from src.core.either import Either
from src.core.contracts import require, ensure


# Branded Types for Testing Architecture
TestExecutionId = str
TestSuiteId = str  
TestRunId = str
QualityReportId = str
TestEnvironmentId = str

def create_test_execution_id() -> TestExecutionId:
    """Create a unique test execution identifier."""
    return f"test_exec_{uuid.uuid4().hex[:12]}"

def create_test_suite_id() -> TestSuiteId:
    """Create a unique test suite identifier."""
    return f"test_suite_{uuid.uuid4().hex[:12]}"

def create_test_run_id() -> TestRunId:
    """Create a unique test run identifier."""
    return f"test_run_{uuid.uuid4().hex[:12]}"

def create_quality_report_id() -> QualityReportId:
    """Create a unique quality report identifier."""
    return f"quality_rpt_{uuid.uuid4().hex[:12]}"

def create_test_environment_id() -> TestEnvironmentId:
    """Create a unique test environment identifier."""
    return f"test_env_{uuid.uuid4().hex[:12]}"


class TestType(Enum):
    """Comprehensive test type classification."""
    # Core Testing Types
    FUNCTIONAL = "functional"
    PERFORMANCE = "performance"
    SECURITY = "security"
    INTEGRATION = "integration"
    
    # Advanced Testing Types
    REGRESSION = "regression"
    LOAD = "load"
    STRESS = "stress"
    ENDURANCE = "endurance"
    
    # Quality Assurance Types
    SMOKE = "smoke"
    SANITY = "sanity"
    COMPATIBILITY = "compatibility"
    USABILITY = "usability"
    
    # Automation-Specific Types
    WORKFLOW = "workflow"
    MACRO_VALIDATION = "macro_validation"
    TRIGGER_TESTING = "trigger_testing"
    CONDITION_TESTING = "condition_testing"


class TestScope(Enum):
    """Test execution scope levels."""
    UNIT = "unit"              # Individual macro/component
    INTEGRATION = "integration" # Multiple macros/components
    SYSTEM = "system"          # Complete automation system
    END_TO_END = "end_to_end"  # Full user workflows
    ACCEPTANCE = "acceptance"   # User acceptance testing


class TestEnvironment(Enum):
    """Test execution environments."""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    SANDBOX = "sandbox"
    ISOLATED = "isolated"


class TestStatus(Enum):
    """Test execution status."""
    PENDING = "pending"
    QUEUED = "queued"
    RUNNING = "running"
    PASSED = "passed"
    FAILED = "failed"
    ERROR = "error"
    SKIPPED = "skipped"
    CANCELLED = "cancelled"
    TIMEOUT = "timeout"


class TestPriority(Enum):
    """Test execution priority levels."""
    CRITICAL = "critical"
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"
    DEBUG = "debug"


class QualityMetric(Enum):
    """Quality assessment metrics."""
    RELIABILITY = "reliability"
    PERFORMANCE = "performance"
    MAINTAINABILITY = "maintainability"
    SECURITY = "security"
    USABILITY = "usability"
    COMPATIBILITY = "compatibility"
    COVERAGE = "coverage"
    STABILITY = "stability"


@dataclass(frozen=True)
class TestConfiguration:
    """Test execution configuration parameters."""
    test_type: TestType
    test_scope: TestScope
    environment: TestEnvironment
    timeout_seconds: int = 300
    retry_count: int = 0
    parallel_execution: bool = False
    resource_limits: Dict[str, Any] = field(default_factory=dict)
    security_level: str = "standard"
    
    def __post_init__(self):
        if not (1 <= self.timeout_seconds <= 7200):
            raise ValueError("Timeout must be between 1 and 7200 seconds")
        if not (0 <= self.retry_count <= 5):
            raise ValueError("Retry count must be between 0 and 5")


@dataclass(frozen=True)
class TestCriteria:
    """Test success/failure criteria."""
    expected_results: Dict[str, Any]
    performance_thresholds: Dict[str, float] = field(default_factory=dict)
    quality_gates: Dict[QualityMetric, float] = field(default_factory=dict)
    custom_validators: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        for metric, threshold in self.quality_gates.items():
            if not (0.0 <= threshold <= 100.0):
                raise ValueError(f"Quality gate threshold for {metric} must be between 0.0 and 100.0")


@dataclass(frozen=True)
class TestAssertion:
    """Individual test assertion definition."""
    assertion_id: str
    assertion_type: str  # equals, contains, greater_than, less_than, matches, custom
    expected_value: Any
    actual_value_path: str  # JSONPath to extract actual value
    description: str
    is_critical: bool = True
    
    def __post_init__(self):
        if not self.assertion_id.strip():
            raise ValueError("Assertion ID cannot be empty")
        if not self.description.strip():
            raise ValueError("Assertion description cannot be empty")


@dataclass(frozen=True)
class TestStep:
    """Individual test execution step."""
    step_id: str
    step_name: str
    step_type: str  # setup, execute, validate, cleanup
    action: str
    parameters: Dict[str, Any]
    assertions: List[TestAssertion] = field(default_factory=list)
    timeout_seconds: int = 60
    retry_on_failure: bool = False
    
    def __post_init__(self):
        if not self.step_id.strip():
            raise ValueError("Step ID cannot be empty")
        if not (1 <= self.timeout_seconds <= 3600):
            raise ValueError("Step timeout must be between 1 and 3600 seconds")


@dataclass(frozen=True)
class AutomationTest:
    """Complete automation test definition."""
    test_id: TestExecutionId
    test_name: str
    description: str
    test_configuration: TestConfiguration
    test_criteria: TestCriteria
    test_steps: List[TestStep]
    setup_steps: List[TestStep] = field(default_factory=list)
    cleanup_steps: List[TestStep] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    dependencies: List[TestExecutionId] = field(default_factory=list)
    
    def __post_init__(self):
        if not self.test_name.strip():
            raise ValueError("Test name cannot be empty")
        if not self.test_steps:
            raise ValueError("Test must have at least one step")


@dataclass(frozen=True)
class TestResult:
    """Test execution result with detailed metrics."""
    test_id: TestExecutionId
    test_run_id: TestRunId
    status: TestStatus
    start_time: datetime
    end_time: datetime
    execution_time_ms: float
    step_results: List[Dict[str, Any]]
    assertions_passed: int
    assertions_failed: int
    performance_metrics: Dict[str, float] = field(default_factory=dict)
    resource_usage: Dict[str, float] = field(default_factory=dict)
    error_details: Optional[str] = None
    artifacts: List[str] = field(default_factory=list)
    
    def __post_init__(self):
        if self.start_time > self.end_time:
            raise ValueError("Start time cannot be after end time")
        if self.execution_time_ms < 0:
            raise ValueError("Execution time cannot be negative")


@dataclass(frozen=True)
class TestSuite:
    """Collection of related automation tests."""
    suite_id: TestSuiteId
    suite_name: str
    description: str
    tests: List[AutomationTest]
    configuration: TestConfiguration
    parallel_execution: bool = False
    max_concurrent_tests: int = 5
    abort_on_failure: bool = False
    
    def __post_init__(self):
        if not self.suite_name.strip():
            raise ValueError("Suite name cannot be empty")
        if not self.tests:
            raise ValueError("Test suite must contain at least one test")
        if not (1 <= self.max_concurrent_tests <= 20):
            raise ValueError("Max concurrent tests must be between 1 and 20")


@dataclass(frozen=True)
class QualityGate:
    """Quality gate definition for test evaluation."""
    gate_id: str
    gate_name: str
    metric: QualityMetric
    threshold: float
    operator: str  # gt, gte, lt, lte, eq
    is_mandatory: bool = True
    description: str = ""
    
    def __post_init__(self):
        if not self.gate_id.strip():
            raise ValueError("Gate ID cannot be empty")
        if not (0.0 <= self.threshold <= 100.0):
            raise ValueError("Threshold must be between 0.0 and 100.0")
        if self.operator not in ["gt", "gte", "lt", "lte", "eq"]:
            raise ValueError("Invalid operator")


@dataclass(frozen=True)
class QualityAssessment:
    """Quality assessment results for test execution."""
    assessment_id: str
    test_run_id: TestRunId
    overall_score: float
    metric_scores: Dict[QualityMetric, float]
    gates_passed: List[QualityGate]
    gates_failed: List[QualityGate]
    recommendations: List[str]
    risk_level: str  # low, medium, high, critical
    
    def __post_init__(self):
        if not (0.0 <= self.overall_score <= 100.0):
            raise ValueError("Overall score must be between 0.0 and 100.0")
        if self.risk_level not in ["low", "medium", "high", "critical"]:
            raise ValueError("Invalid risk level")


@dataclass(frozen=True)
class RegressionDetection:
    """Regression detection configuration and results."""
    detection_id: str
    baseline_run_id: TestRunId
    current_run_id: TestRunId
    detection_sensitivity: str  # low, medium, high
    metrics_to_compare: List[str]
    threshold_percentage: float = 5.0  # % degradation threshold
    regressions_found: List[Dict[str, Any]] = field(default_factory=list)
    improvements_found: List[Dict[str, Any]] = field(default_factory=list)
    
    def __post_init__(self):
        if not (0.1 <= self.threshold_percentage <= 50.0):
            raise ValueError("Threshold percentage must be between 0.1 and 50.0")
        if self.detection_sensitivity not in ["low", "medium", "high"]:
            raise ValueError("Invalid detection sensitivity")


class TestingArchitectureError(Exception):
    """Base exception for testing architecture errors."""
    
    def __init__(self, message: str, error_code: str = "TESTING_ERROR"):
        super().__init__(message)
        self.error_code = error_code
        self.timestamp = datetime.now(UTC)


class TestExecutionError(TestingArchitectureError):
    """Test execution specific errors."""
    
    @classmethod
    def timeout_exceeded(cls, test_id: str, timeout: int) -> TestExecutionError:
        return cls(f"Test {test_id} exceeded timeout of {timeout} seconds", "TEST_TIMEOUT")
    
    @classmethod
    def assertion_failed(cls, test_id: str, assertion: str) -> TestExecutionError:
        return cls(f"Assertion failed in test {test_id}: {assertion}", "ASSERTION_FAILED")
    
    @classmethod
    def setup_failed(cls, test_id: str, error: str) -> TestExecutionError:
        return cls(f"Setup failed for test {test_id}: {error}", "SETUP_FAILED")


class QualityGateError(TestingArchitectureError):
    """Quality gate validation errors."""
    
    @classmethod
    def gate_failed(cls, gate_name: str, actual: float, expected: float) -> QualityGateError:
        return cls(f"Quality gate '{gate_name}' failed: {actual} does not meet threshold {expected}", "GATE_FAILED")


class RegressionError(TestingArchitectureError):
    """Regression detection errors."""
    
    @classmethod
    def regression_detected(cls, metric: str, degradation: float) -> RegressionError:
        return cls(f"Regression detected in {metric}: {degradation}% degradation", "REGRESSION_DETECTED")


# Utility Functions for Test Management

@require(lambda test_name: len(test_name.strip()) > 0)
@require(lambda test_type: isinstance(test_type, TestType))
def create_simple_test(
    test_name: str,
    test_type: TestType,
    target_macro_id: Optional[MacroId] = None,
    timeout_seconds: int = 300
) -> AutomationTest:
    """Create a simple automation test with minimal configuration."""
    test_id = create_test_execution_id()
    
    configuration = TestConfiguration(
        test_type=test_type,
        test_scope=TestScope.UNIT,
        environment=TestEnvironment.SANDBOX,
        timeout_seconds=timeout_seconds
    )
    
    criteria = TestCriteria(
        expected_results={"status": "success"},
        quality_gates={QualityMetric.RELIABILITY: 95.0}
    )
    
    # Basic test step for macro execution
    test_step = TestStep(
        step_id=f"step_{uuid.uuid4().hex[:8]}",
        step_name="Execute Target",
        step_type="execute",
        action="execute_macro" if target_macro_id else "validate_system",
        parameters={"macro_id": target_macro_id} if target_macro_id else {},
        assertions=[
            TestAssertion(
                assertion_id="success_check",
                assertion_type="equals",
                expected_value="success",
                actual_value_path="$.status",
                description="Verify successful execution"
            )
        ]
    )
    
    return AutomationTest(
        test_id=test_id,
        test_name=test_name,
        description=f"Automated {test_type.value} test for {test_name}",
        test_configuration=configuration,
        test_criteria=criteria,
        test_steps=[test_step]
    )


@require(lambda suite_name: len(suite_name.strip()) > 0)
@require(lambda tests: len(tests) > 0)
def create_test_suite(
    suite_name: str,
    tests: List[AutomationTest],
    parallel_execution: bool = False,
    max_concurrent: int = 5
) -> TestSuite:
    """Create a test suite from a collection of tests."""
    suite_id = create_test_suite_id()
    
    # Use the most restrictive configuration from all tests
    configuration = TestConfiguration(
        test_type=TestType.INTEGRATION,  # Suite-level default
        test_scope=TestScope.INTEGRATION,
        environment=TestEnvironment.SANDBOX,
        timeout_seconds=max(test.test_configuration.timeout_seconds for test in tests),
        parallel_execution=parallel_execution
    )
    
    return TestSuite(
        suite_id=suite_id,
        suite_name=suite_name,
        description=f"Test suite containing {len(tests)} automation tests",
        tests=tests,
        configuration=configuration,
        parallel_execution=parallel_execution,
        max_concurrent_tests=max_concurrent
    )


def create_quality_gates(
    reliability_threshold: float = 95.0,
    performance_threshold: float = 90.0,
    security_threshold: float = 100.0
) -> List[QualityGate]:
    """Create standard quality gates for automation testing."""
    return [
        QualityGate(
            gate_id="reliability_gate",
            gate_name="Reliability Gate",
            metric=QualityMetric.RELIABILITY,
            threshold=reliability_threshold,
            operator="gte",
            is_mandatory=True,
            description="Ensure automation reliability meets standards"
        ),
        QualityGate(
            gate_id="performance_gate",
            gate_name="Performance Gate",
            metric=QualityMetric.PERFORMANCE,
            threshold=performance_threshold,
            operator="gte",
            is_mandatory=True,
            description="Ensure automation performance meets requirements"
        ),
        QualityGate(
            gate_id="security_gate",
            gate_name="Security Gate",
            metric=QualityMetric.SECURITY,
            threshold=security_threshold,
            operator="gte",
            is_mandatory=True,
            description="Ensure automation security compliance"
        )
    ]


@ensure(lambda result: 0.0 <= result <= 100.0)
def calculate_quality_score(metric_scores: Dict[QualityMetric, float]) -> float:
    """Calculate overall quality score from individual metrics."""
    if not metric_scores:
        return 0.0
    
    # Weighted average of quality metrics
    weights = {
        QualityMetric.RELIABILITY: 0.25,
        QualityMetric.PERFORMANCE: 0.20,
        QualityMetric.SECURITY: 0.25,
        QualityMetric.MAINTAINABILITY: 0.15,
        QualityMetric.COVERAGE: 0.15
    }
    
    total_weight = 0.0
    weighted_sum = 0.0
    
    for metric, score in metric_scores.items():
        weight = weights.get(metric, 0.1)  # Default weight for unknown metrics
        weighted_sum += score * weight
        total_weight += weight
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0


def determine_risk_level(quality_score: float, failed_gates: List[QualityGate]) -> str:
    """Determine risk level based on quality score and failed gates."""
    if quality_score >= 90.0 and not failed_gates:
        return "low"
    elif quality_score >= 75.0 and len(failed_gates) <= 1:
        return "medium"
    elif quality_score >= 50.0 or any(gate.is_mandatory for gate in failed_gates):
        return "high"
    else:
        return "critical"