"""Comprehensive tests for src/intelligence/workflow_analyzer.py - MASSIVE 436 statements coverage.

🚨 CRITICAL COVERAGE ENFORCEMENT: Phase 8 targeting highest-impact zero-coverage modules.
This test covers src/intelligence/workflow_analyzer.py (436 statements - 5th HIGHEST IMPACT) to achieve
significant progress toward mandatory 95% coverage threshold.

Coverage Focus: WorkflowAnalyzer class, workflow analysis engine, pattern recognition,
performance prediction, quality assessment, optimization recommendations, and all intelligent workflow processing.
"""

from collections import defaultdict
from datetime import timedelta
from unittest.mock import Mock, patch

import pytest
from src.core.errors import ValidationError
from src.core.workflow_intelligence import (
    IntelligenceLevel,
    OptimizationGoal,
    PatternType,
    WorkflowComplexity,
    WorkflowComponent,
    WorkflowPattern,
)
from src.intelligence.workflow_analyzer import AnalysisMetrics, WorkflowAnalyzer


class TestAnalysisMetrics:
    """Comprehensive tests for AnalysisMetrics dataclass."""

    def test_analysis_metrics_creation_success(self):
        """Test successful AnalysisMetrics creation."""
        metrics = AnalysisMetrics(
            total_components=5,
            unique_component_types=3,
            dependency_depth=2,
            cyclic_dependencies=False,
            resource_conflicts=["file_conflict"],
            performance_bottlenecks=["slow_operation"],
            reliability_concerns=["low_reliability_component"],
        )

        assert metrics.total_components == 5
        assert metrics.unique_component_types == 3
        assert metrics.dependency_depth == 2
        assert metrics.cyclic_dependencies is False
        assert metrics.resource_conflicts == ["file_conflict"]
        assert metrics.performance_bottlenecks == ["slow_operation"]
        assert metrics.reliability_concerns == ["low_reliability_component"]

    def test_analysis_metrics_empty_lists(self):
        """Test AnalysisMetrics with empty lists."""
        metrics = AnalysisMetrics(
            total_components=0,
            unique_component_types=0,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        assert metrics.total_components == 0
        assert len(metrics.resource_conflicts) == 0
        assert len(metrics.performance_bottlenecks) == 0
        assert len(metrics.reliability_concerns) == 0

    def test_analysis_metrics_with_cyclic_dependencies(self):
        """Test AnalysisMetrics with cyclic dependencies detected."""
        metrics = AnalysisMetrics(
            total_components=3,
            unique_component_types=2,
            dependency_depth=5,
            cyclic_dependencies=True,
            resource_conflicts=["app_conflict", "file_conflict"],
            performance_bottlenecks=["bottleneck_1", "bottleneck_2"],
            reliability_concerns=["reliability_issue"],
        )

        assert metrics.cyclic_dependencies is True
        assert metrics.dependency_depth == 5
        assert len(metrics.resource_conflicts) == 2
        assert len(metrics.performance_bottlenecks) == 2


class TestWorkflowAnalyzerInitialization:
    """Comprehensive tests for WorkflowAnalyzer initialization."""

    def test_workflow_analyzer_default_initialization(self):
        """Test WorkflowAnalyzer initialization with default configuration."""
        analyzer = WorkflowAnalyzer()

        assert analyzer.config == {}
        assert analyzer.analysis_history == {}
        # Pattern library is initialized with default patterns
        assert len(analyzer.pattern_library) >= 2  # Has default patterns
        assert isinstance(analyzer.optimization_templates, defaultdict)
        assert analyzer.performance_benchmarks["simple_workflow_ms"] == 100
        assert analyzer.quality_thresholds["excellent"] == 0.9
        assert analyzer.analysis_stats["total_analyses"] == 0

    def test_workflow_analyzer_custom_config_initialization(self):
        """Test WorkflowAnalyzer initialization with custom configuration."""
        custom_config = {
            "max_analysis_depth": 5,
            "enable_ml_patterns": True,
            "performance_threshold": 0.8,
        }
        analyzer = WorkflowAnalyzer(intelligence_config=custom_config)

        assert analyzer.config == custom_config
        assert analyzer.config["max_analysis_depth"] == 5
        assert analyzer.config["enable_ml_patterns"] is True

    def test_workflow_analyzer_pattern_library_initialization(self):
        """Test that pattern library is properly initialized."""
        analyzer = WorkflowAnalyzer()

        # Check that common patterns are initialized
        assert "sequential_processing" in analyzer.pattern_library
        assert "error_handling" in analyzer.pattern_library

        # Verify sequential processing pattern
        seq_pattern = analyzer.pattern_library["sequential_processing"]
        assert seq_pattern.name == "Sequential Processing"
        assert seq_pattern.pattern_type == PatternType.EFFICIENCY
        assert seq_pattern.usage_frequency == 0.7

        # Verify error handling pattern
        error_pattern = analyzer.pattern_library["error_handling"]
        assert error_pattern.name == "Comprehensive Error Handling"
        assert error_pattern.pattern_type == PatternType.BEST_PRACTICE
        assert error_pattern.effectiveness_score == 0.95

    def test_workflow_analyzer_performance_benchmarks(self):
        """Test performance benchmarks initialization."""
        analyzer = WorkflowAnalyzer()

        expected_benchmarks = {
            "simple_workflow_ms": 100,
            "intermediate_workflow_ms": 500,
            "advanced_workflow_ms": 2000,
            "expert_workflow_ms": 5000,
        }

        assert analyzer.performance_benchmarks == expected_benchmarks

    def test_workflow_analyzer_quality_thresholds(self):
        """Test quality thresholds initialization."""
        analyzer = WorkflowAnalyzer()

        expected_thresholds = {
            "excellent": 0.9,
            "good": 0.75,
            "fair": 0.6,
            "poor": 0.4,
        }

        assert analyzer.quality_thresholds == expected_thresholds

    def test_workflow_analyzer_analysis_stats_initialization(self):
        """Test analysis statistics initialization."""
        analyzer = WorkflowAnalyzer()

        expected_stats = {
            "total_analyses": 0,
            "patterns_identified": 0,
            "optimizations_suggested": 0,
            "avg_analysis_time_ms": 0.0,
            "quality_score_average": 0.0,
        }

        assert analyzer.analysis_stats == expected_stats


class TestWorkflowAnalyzerPerformanceAnalysis:
    """Comprehensive tests for WorkflowAnalyzer performance analysis methods."""

    def test_analyze_performance_success(self):
        """Test successful performance analysis."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "steps": [
                {"name": "Step 1", "duration": 100},
                {"name": "Step 2", "duration": 150},
                {"name": "Step 3", "duration": 120},
            ],
            "total_duration": 370,
        }

        result = analyzer._analyze_performance(workflow_data)

        assert "efficiency" in result
        assert "step_count" in result
        assert "total_duration" in result
        assert "average_step_duration" in result
        assert "bottlenecks" in result
        assert "insights" in result
        assert result["step_count"] == 3
        assert result["total_duration"] == 370
        assert result["average_step_duration"] == pytest.approx(123.33, rel=1e-2)

    def test_analyze_performance_no_steps(self):
        """Test performance analysis with no steps."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {"steps": [], "total_duration": 0}

        result = analyzer._analyze_performance(workflow_data)

        assert result["efficiency"] == 0.0
        assert result["analysis"] == "No steps found in workflow"

    def test_analyze_performance_bottleneck_detection(self):
        """Test performance bottleneck detection."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "steps": [
                {"name": "Fast Step", "duration": 50},
                {"name": "Slow Step", "duration": 500},  # Much slower than average
                {"name": "Normal Step", "duration": 100},
            ],
            "total_duration": 650,
        }

        result = analyzer._analyze_performance(workflow_data)

        assert len(result["bottlenecks"]) > 0
        assert "Slow Step" in result["bottlenecks"][0]

    def test_analyze_performance_excellent_efficiency(self):
        """Test performance analysis with excellent efficiency."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "steps": [
                {"name": "Step 1", "duration": 10},
                {"name": "Step 2", "duration": 10},
                {"name": "Step 3", "duration": 10},
            ],
            "total_duration": 30,
        }

        result = analyzer._analyze_performance(workflow_data)

        assert result["efficiency"] > 0.8
        assert any(
            "excellent performance" in insight.lower() for insight in result["insights"]
        )

    def test_analyze_performance_poor_efficiency(self):
        """Test performance analysis with poor efficiency."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "steps": [
                {"name": "Step 1", "duration": 1000},
                {"name": "Step 2", "duration": 2000},
                {"name": "Step 3", "duration": 3000},
            ],
            "total_duration": 6000,
        }

        result = analyzer._analyze_performance(workflow_data)

        assert result["efficiency"] < 0.6
        assert any(
            "optimization" in insight.lower() for insight in result["insights"]
        )

    def test_analyze_performance_error_handling(self):
        """Test performance analysis error handling."""
        analyzer = WorkflowAnalyzer()

        # Mock logger to capture error
        with patch.object(analyzer.logger, "error") as mock_logger:
            # Pass invalid workflow data that will cause an exception
            result = analyzer._analyze_performance(None)

            assert result["efficiency"] == 0.0
            assert "error" in result
            assert result["analysis"] == "Performance analysis failed"
            mock_logger.assert_called_once()

    def test_analyze_performance_zero_durations(self):
        """Test performance analysis with zero durations."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "steps": [
                {"name": "Step 1", "duration": 0},
                {"name": "Step 2", "duration": 0},
            ],
            "total_duration": 0,
        }

        result = analyzer._analyze_performance(workflow_data)

        assert result["efficiency"] == 0.5  # Default for zero duration
        assert result["total_duration"] == 0
        assert result["average_step_duration"] == 0


class TestWorkflowAnalyzerComponentExtraction:
    """Comprehensive tests for workflow component extraction methods."""

    def test_extract_workflow_components_direct_format(self):
        """Test component extraction from direct component format."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "components": [
                {
                    "component_id": "comp_001",
                    "component_type": "action",
                    "name": "Test Action",
                    "description": "Test action component",
                    "parameters": {"param1": "value1"},
                    "dependencies": ["comp_002"],
                    "execution_time_ms": 500,
                    "reliability_score": 0.9,
                    "complexity_score": 0.3,
                }
            ]
        }

        components = analyzer._extract_workflow_components(workflow_data)

        assert len(components) == 1
        component = components[0]
        assert component.component_id == "comp_001"
        assert component.component_type == "action"
        assert component.name == "Test Action"
        assert component.parameters["param1"] == "value1"

    def test_extract_workflow_components_action_format(self):
        """Test component extraction from action-based format."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "actions": [
                {
                    "id": "action_001",
                    "name": "Test Action",
                    "description": "Test action",
                    "parameters": {"key": "value"},
                    "dependencies": [],
                    "execution_time_ms": 300,
                    "reliability": 0.95,
                    "complexity": 0.2,
                },
                {
                    "name": "Action Without ID",
                    "description": "Action with generated ID",
                    "execution_time_ms": 400,
                },
            ]
        }

        components = analyzer._extract_workflow_components(workflow_data)

        assert len(components) == 2
        assert components[0].component_id == "action_001"
        assert components[0].name == "Test Action"
        assert components[1].component_id == "action_1"  # Generated ID
        assert components[1].name == "Action Without ID"

    def test_extract_workflow_components_empty_workflow(self):
        """Test component extraction from empty workflow."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {}

        components = analyzer._extract_workflow_components(workflow_data)

        assert len(components) == 0

    def test_create_component_from_data_success(self):
        """Test successful component creation from data."""
        analyzer = WorkflowAnalyzer()
        comp_data = {
            "component_id": "test_comp",
            "component_type": "condition",
            "name": "Test Component",
            "description": "Test component description",
            "parameters": {"test_param": "test_value"},
            "dependencies": ["dep1", "dep2"],
            "execution_time_ms": 250,
            "reliability_score": 0.85,
            "complexity_score": 0.4,
        }

        component = analyzer._create_component_from_data(comp_data)

        assert component is not None
        assert component.component_id == "test_comp"
        assert component.component_type == "condition"
        assert component.name == "Test Component"
        assert component.description == "Test component description"
        assert component.reliability_score == 0.85
        assert component.complexity_score == 0.4

    def test_create_component_from_data_minimal(self):
        """Test component creation with minimal data."""
        analyzer = WorkflowAnalyzer()
        comp_data = {}

        component = analyzer._create_component_from_data(comp_data)

        assert component is not None
        assert component.component_type == "action"  # Default
        assert component.name == "Component"  # Default
        assert component.reliability_score == 0.9  # Default
        assert component.complexity_score == 0.3  # Default

    def test_create_component_from_data_error_handling(self):
        """Test component creation error handling."""
        analyzer = WorkflowAnalyzer()

        with patch.object(analyzer.logger, "warning") as mock_logger:
            # Pass data that will cause an exception during component creation
            component = analyzer._create_component_from_data({"invalid": True})

            # Should handle gracefully and return None
            assert component is not None  # Actually succeeds with defaults
            # But let's test with truly invalid data
            with patch(
                "src.intelligence.workflow_analyzer.WorkflowComponent",
                side_effect=Exception("Test error"),
            ):
                component = analyzer._create_component_from_data({})
                assert component is None
                mock_logger.assert_called_once()


class TestWorkflowAnalyzerDependencyAnalysis:
    """Comprehensive tests for dependency analysis methods."""

    def test_calculate_dependency_depth_linear(self):
        """Test dependency depth calculation for linear dependencies."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": ["B"], "B": ["C"], "C": []}

        depth = analyzer._calculate_dependency_depth(dependency_graph)

        assert depth == 2  # A -> B -> C (depth 2)

    def test_calculate_dependency_depth_branching(self):
        """Test dependency depth calculation for branching dependencies."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": ["B", "C"], "B": ["D"], "C": ["E"], "D": [], "E": []}

        depth = analyzer._calculate_dependency_depth(dependency_graph)

        assert depth == 2  # A -> B -> D and A -> C -> E (both depth 2)

    def test_calculate_dependency_depth_complex(self):
        """Test dependency depth calculation for complex dependencies."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {
            "A": ["B"],
            "B": ["C"],
            "C": ["D"],
            "D": ["E"],
            "E": [],
            "F": ["G"],
            "G": [],
        }

        depth = analyzer._calculate_dependency_depth(dependency_graph)

        assert depth == 4  # A -> B -> C -> D -> E (depth 4)

    def test_calculate_dependency_depth_no_dependencies(self):
        """Test dependency depth calculation with no dependencies."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": [], "B": [], "C": []}

        depth = analyzer._calculate_dependency_depth(dependency_graph)

        assert depth == 0

    def test_detect_cyclic_dependencies_no_cycle(self):
        """Test cyclic dependency detection with no cycles."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": ["B"], "B": ["C"], "C": []}

        has_cycle = analyzer._detect_cyclic_dependencies(dependency_graph)

        assert has_cycle is False

    def test_detect_cyclic_dependencies_simple_cycle(self):
        """Test cyclic dependency detection with simple cycle."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": ["B"], "B": ["C"], "C": ["A"]}

        has_cycle = analyzer._detect_cyclic_dependencies(dependency_graph)

        assert has_cycle is True

    def test_detect_cyclic_dependencies_self_cycle(self):
        """Test cyclic dependency detection with self-referencing cycle."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {"A": ["A"], "B": []}

        has_cycle = analyzer._detect_cyclic_dependencies(dependency_graph)

        assert has_cycle is True

    def test_detect_cyclic_dependencies_complex_cycle(self):
        """Test cyclic dependency detection with complex cycle."""
        analyzer = WorkflowAnalyzer()
        dependency_graph = {
            "A": ["B"],
            "B": ["C", "D"],
            "C": ["E"],
            "D": ["F"],
            "E": [],
            "F": ["B"],  # Creates cycle B -> D -> F -> B
        }

        has_cycle = analyzer._detect_cyclic_dependencies(dependency_graph)

        assert has_cycle is True


class TestWorkflowAnalyzerResourceConflictDetection:
    """Comprehensive tests for resource conflict detection."""

    def test_detect_resource_conflicts_file_conflicts(self):
        """Test detection of file resource conflicts."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="File Reader 1",
                description="Reads file",
                parameters={"file_path": "/tmp/test.txt"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="File Reader 2",
                description="Also reads file",
                parameters={"file_path": "/tmp/test.txt"},  # Same file!
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        conflicts = analyzer._detect_resource_conflicts(components)

        assert len(conflicts) == 1
        assert "File resource conflict" in conflicts[0]
        assert "/tmp/test.txt" in conflicts[0]
        assert "comp1" in conflicts[0] and "comp2" in conflicts[0]

    def test_detect_resource_conflicts_application_conflicts(self):
        """Test detection of application resource conflicts."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="App User 1",
                description="Uses application",
                parameters={"application": "calculator"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="App User 2",
                description="Also uses application",
                parameters={"application": "calculator"},  # Same app!
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        conflicts = analyzer._detect_resource_conflicts(components)

        assert len(conflicts) == 1
        assert "Application resource conflict" in conflicts[0]
        assert "calculator" in conflicts[0]

    def test_detect_resource_conflicts_multiple_files(self):
        """Test detection with multiple file paths."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Multi File User",
                description="Uses multiple files",
                parameters={"file_paths": ["/tmp/file1.txt", "/tmp/file2.txt"]},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="Single File User",
                description="Uses one file",
                parameters={"file_path": "/tmp/file1.txt"},  # Conflicts with first
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        conflicts = analyzer._detect_resource_conflicts(components)

        assert len(conflicts) == 1
        assert "/tmp/file1.txt" in conflicts[0]

    def test_detect_resource_conflicts_no_conflicts(self):
        """Test resource conflict detection with no conflicts."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Component 1",
                description="Uses unique resources",
                parameters={"file_path": "/tmp/file1.txt"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="Component 2",
                description="Uses different resources",
                parameters={"file_path": "/tmp/file2.txt"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        conflicts = analyzer._detect_resource_conflicts(components)

        assert len(conflicts) == 0


class TestWorkflowAnalyzerPerformanceBottlenecks:
    """Comprehensive tests for performance bottleneck identification."""

    def test_identify_performance_bottlenecks_slow_components(self):
        """Test identification of slow components as bottlenecks."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="fast_comp1",
                component_type="action",
                name="Fast Component 1",
                description="Fast operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="fast_comp2",
                component_type="action",
                name="Fast Component 2",
                description="Fast operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="fast_comp3",
                component_type="action",
                name="Fast Component 3",
                description="Fast operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="slow_comp",
                component_type="action",
                name="Slow Component",
                description="Slow operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(seconds=5),  # 5s with avg ~0.1s, 3x avg = 0.3s, so 5s > 0.3s triggers
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        bottlenecks = analyzer._identify_performance_bottlenecks(components)

        assert len(bottlenecks) > 0
        assert any("Slow Component" in bottleneck for bottleneck in bottlenecks)

    def test_identify_performance_bottlenecks_parallelization_opportunity(self):
        """Test identification of parallelization opportunities."""
        analyzer = WorkflowAnalyzer()
        # Create many independent actions (more than DEFAULT_RETRY_COUNT)
        components = []
        for i in range(6):  # More than DEFAULT_RETRY_COUNT (3)
            components.append(
                WorkflowComponent(
                    component_id=f"action_{i}",
                    component_type="action",
                    name=f"Independent Action {i}",
                    description="Independent action",
                    parameters={},
                    dependencies=[],  # No dependencies = independent
                    estimated_execution_time=timedelta(milliseconds=100),
                    reliability_score=0.9,
                    complexity_score=0.2,
                )
            )

        bottlenecks = analyzer._identify_performance_bottlenecks(components)

        assert len(bottlenecks) > 0
        assert any("parallelization opportunity" in bottleneck for bottleneck in bottlenecks)

    def test_identify_performance_bottlenecks_no_bottlenecks(self):
        """Test bottleneck identification with no bottlenecks."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Component 1",
                description="Normal component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="Component 2",
                description="Another normal component",
                parameters={},
                dependencies=["comp1"],  # Has dependency
                estimated_execution_time=timedelta(milliseconds=150),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        bottlenecks = analyzer._identify_performance_bottlenecks(components)

        # Should have no bottlenecks (times are similar, not many independent actions)
        assert len(bottlenecks) == 0


class TestWorkflowAnalyzerReliabilityConcerns:
    """Comprehensive tests for reliability concern identification."""

    def test_identify_reliability_concerns_low_reliability_components(self):
        """Test identification of low reliability components."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="reliable_comp",
                component_type="action",
                name="Reliable Component",
                description="Highly reliable",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.95,  # High reliability
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="unreliable_comp",
                component_type="action",
                name="Unreliable Component",
                description="Low reliability",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.6,  # Low reliability (<0.8)
                complexity_score=0.2,
            ),
        ]

        concerns = analyzer._identify_reliability_concerns(components)

        assert len(concerns) > 0
        assert any("Unreliable Component" in concern for concern in concerns)

    def test_identify_reliability_concerns_missing_error_handling(self):
        """Test identification of missing error handling."""
        analyzer = WorkflowAnalyzer()
        # Create workflow with many components but no conditions (error handling)
        components = []
        for i in range(6):  # More than DEFAULT_RETRY_COUNT
            components.append(
                WorkflowComponent(
                    component_id=f"action_{i}",
                    component_type="action",  # Only actions, no conditions
                    name=f"Action {i}",
                    description="Action without error handling",
                    parameters={},
                    dependencies=[],
                    estimated_execution_time=timedelta(milliseconds=100),
                    reliability_score=0.9,
                    complexity_score=0.2,
                )
            )

        concerns = analyzer._identify_reliability_concerns(components)

        assert len(concerns) > 0
        assert any("error handling" in concern.lower() for concern in concerns)

    def test_identify_reliability_concerns_none_found(self):
        """Test reliability concern identification with no concerns."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="action_comp",
                component_type="action",
                name="Reliable Action",
                description="Reliable action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.95,  # High reliability
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="condition_comp",
                component_type="condition",
                name="Error Handler",
                description="Error handling condition",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=50),
                reliability_score=0.98,
                complexity_score=0.1,
            ),
        ]

        concerns = analyzer._identify_reliability_concerns(components)

        # Should have no concerns (good reliability, has error handling)
        assert len(concerns) == 0


class TestWorkflowAnalyzerPatternRecognition:
    """Comprehensive tests for workflow pattern recognition."""

    def test_matches_sequential_pattern_true(self):
        """Test sequential pattern matching - should match."""
        analyzer = WorkflowAnalyzer()
        # Create components with mostly no dependencies (sequential)
        components = []
        for i in range(5):
            components.append(
                WorkflowComponent(
                    component_id=f"seq_{i}",
                    component_type="action",
                    name=f"Sequential Action {i}",
                    description="Sequential action",
                    parameters={},
                    dependencies=[],  # No dependencies = sequential
                    estimated_execution_time=timedelta(milliseconds=100),
                    reliability_score=0.9,
                    complexity_score=0.2,
                )
            )

        matches = analyzer._matches_sequential_pattern(components)

        assert matches is True

    def test_matches_sequential_pattern_false_too_few_components(self):
        """Test sequential pattern matching - too few components."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Single Action",
                description="Not enough for pattern",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            )
        ]

        matches = analyzer._matches_sequential_pattern(components)

        assert matches is False

    def test_matches_sequential_pattern_false_too_many_dependencies(self):
        """Test sequential pattern matching - too many dependencies."""
        analyzer = WorkflowAnalyzer()
        components = []
        for i in range(5):
            dependencies = [f"dep_{j}" for j in range(i)]  # Each has more dependencies
            components.append(
                WorkflowComponent(
                    component_id=f"dep_{i}",
                    component_type="action",
                    name=f"Dependent Action {i}",
                    description="Has dependencies",
                    parameters={},
                    dependencies=dependencies,
                    estimated_execution_time=timedelta(milliseconds=100),
                    reliability_score=0.9,
                    complexity_score=0.2,
                )
            )

        matches = analyzer._matches_sequential_pattern(components)

        assert matches is False  # Too many dependencies, not sequential

    def test_matches_error_handling_pattern_true(self):
        """Test error handling pattern matching - should match."""
        analyzer = WorkflowAnalyzer()
        components = [
            # 3 actions
            WorkflowComponent(
                component_id="action1",
                component_type="action",
                name="Action 1",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="action2",
                component_type="action",
                name="Action 2",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="action3",
                component_type="action",
                name="Action 3",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            # 1 condition (good ratio: 1/3 = 0.33 >= 0.25)
            WorkflowComponent(
                component_id="condition1",
                component_type="condition",
                name="Error Check",
                description="Error handling condition",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=50),
                reliability_score=0.9,
                complexity_score=0.1,
            ),
        ]

        matches = analyzer._matches_error_handling_pattern(components)

        assert matches is True

    def test_matches_error_handling_pattern_false(self):
        """Test error handling pattern matching - should not match."""
        analyzer = WorkflowAnalyzer()
        components = [
            # Many actions but no conditions
            WorkflowComponent(
                component_id="action1",
                component_type="action",
                name="Action 1",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="action2",
                component_type="action",
                name="Action 2",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="action3",
                component_type="action",
                name="Action 3",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="action4",
                component_type="action",
                name="Action 4",
                description="Action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        matches = analyzer._matches_error_handling_pattern(components)

        assert matches is False  # No conditions, so no error handling


class TestWorkflowAnalyzerQualityAssessment:
    """Comprehensive tests for workflow quality assessment."""

    @pytest.mark.asyncio
    async def test_assess_workflow_quality_high_quality(self):
        """Test quality assessment for high-quality workflow."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="High Quality Component",
                description="Well-designed component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.95,  # High reliability
                complexity_score=0.2,  # Low complexity
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="condition",
                name="Error Handler",
                description="Error handling",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=50),
                reliability_score=0.98,
                complexity_score=0.1,
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=2,
            unique_component_types=2,
            dependency_depth=1,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        quality_score = await analyzer._assess_workflow_quality(components, metrics)

        assert quality_score >= 0.8  # Should be high quality

    @pytest.mark.asyncio
    async def test_assess_workflow_quality_low_quality(self):
        """Test quality assessment for low-quality workflow."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Low Quality Component",
                description="Poorly designed",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.4,  # Low reliability
                complexity_score=0.9,  # High complexity
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=10,  # Very deep dependencies
            cyclic_dependencies=True,  # Has cycles
            resource_conflicts=["conflict1", "conflict2"],
            performance_bottlenecks=["bottleneck1"],
            reliability_concerns=["concern1"],
        )

        quality_score = await analyzer._assess_workflow_quality(components, metrics)

        assert quality_score <= 0.6  # Should be low quality (allowing some tolerance)

    @pytest.mark.asyncio
    async def test_assess_workflow_quality_empty_components(self):
        """Test quality assessment with empty components."""
        analyzer = WorkflowAnalyzer()
        components = []

        metrics = AnalysisMetrics(
            total_components=0,
            unique_component_types=0,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        quality_score = await analyzer._assess_workflow_quality(components, metrics)

        assert 0.0 <= quality_score <= 1.0  # Should be valid score


class TestWorkflowAnalyzerComplexityAnalysis:
    """Comprehensive tests for complexity analysis."""

    @pytest.mark.asyncio
    async def test_analyze_complexity_simple_workflow(self):
        """Test complexity analysis for simple workflow."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="simple_comp",
                component_type="action",
                name="Simple Component",
                description="Simple action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.1,  # Low complexity
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        # Mock calculate_workflow_complexity_score to return low complexity
        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.2,
        ):
            complexity_analysis = await analyzer._analyze_complexity(components, metrics)

        assert complexity_analysis["complexity_level"] == WorkflowComplexity.SIMPLE.value
        assert complexity_analysis["overall_score"] == 0.2
        assert complexity_analysis["component_count"] == 1

    @pytest.mark.asyncio
    async def test_analyze_complexity_expert_workflow(self):
        """Test complexity analysis for expert-level workflow."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="complex_comp",
                component_type="action",
                name="Complex Component",
                description="Very complex action",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.9,  # High complexity
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=10,
            cyclic_dependencies=True,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        # Mock calculate_workflow_complexity_score to return high complexity
        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.9,
        ):
            complexity_analysis = await analyzer._analyze_complexity(components, metrics)

        assert complexity_analysis["complexity_level"] == WorkflowComplexity.EXPERT.value
        assert complexity_analysis["overall_score"] == 0.9
        assert complexity_analysis["has_cycles"] is True


class TestWorkflowAnalyzerPerformancePrediction:
    """Comprehensive tests for performance prediction."""

    @pytest.mark.asyncio
    async def test_predict_performance_success(self):
        """Test successful performance prediction."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Component 1",
                description="Test component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(seconds=1),
                reliability_score=0.9,
                complexity_score=0.3,
            ),
            WorkflowComponent(
                component_id="comp2",
                component_type="action",
                name="Component 2",
                description="Test component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(seconds=2),
                reliability_score=0.8,
                complexity_score=0.4,
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=2,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        # Mock estimate_workflow_execution_time
        with patch(
            "src.intelligence.workflow_analyzer.estimate_workflow_execution_time",
            return_value=timedelta(seconds=3),
        ):
            prediction = await analyzer._predict_performance(components, metrics)

        assert prediction["estimated_execution_time_seconds"] == 3.0
        assert prediction["estimated_throughput_per_hour"] == 1200.0  # 3600/3
        assert "estimated_cpu_usage_percent" in prediction
        assert "estimated_memory_usage_mb" in prediction
        assert "predicted_success_rate" in prediction

    @pytest.mark.asyncio
    async def test_predict_performance_zero_execution_time(self):
        """Test performance prediction with zero execution time."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="fast_comp",
                component_type="action",
                name="Instant Component",
                description="Zero time component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(seconds=0),
                reliability_score=0.9,
                complexity_score=0.1,
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        # Mock estimate_workflow_execution_time to return zero
        with patch(
            "src.intelligence.workflow_analyzer.estimate_workflow_execution_time",
            return_value=timedelta(seconds=0),
        ):
            prediction = await analyzer._predict_performance(components, metrics)

        assert prediction["estimated_execution_time_seconds"] == 0.0
        assert prediction["estimated_throughput_per_hour"] == 3600  # Very fast


class TestWorkflowAnalyzerStatisticsAndUtilities:
    """Comprehensive tests for statistics and utility methods."""

    def test_update_analysis_stats(self):
        """Test analysis statistics update."""
        analyzer = WorkflowAnalyzer()

        # Initial stats should be zero
        assert analyzer.analysis_stats["total_analyses"] == 0

        # Update stats
        analyzer._update_analysis_stats(
            analysis_time=100.0, quality_score=0.8, patterns_found=2, optimizations_count=3
        )

        assert analyzer.analysis_stats["total_analyses"] == 1
        assert analyzer.analysis_stats["patterns_identified"] == 2
        assert analyzer.analysis_stats["optimizations_suggested"] == 3
        assert analyzer.analysis_stats["avg_analysis_time_ms"] == 100.0
        assert analyzer.analysis_stats["quality_score_average"] == 0.8

        # Update again to test averaging
        analyzer._update_analysis_stats(
            analysis_time=200.0, quality_score=0.6, patterns_found=1, optimizations_count=2
        )

        assert analyzer.analysis_stats["total_analyses"] == 2
        assert analyzer.analysis_stats["patterns_identified"] == 3  # 2 + 1
        assert analyzer.analysis_stats["optimizations_suggested"] == 5  # 3 + 2
        assert analyzer.analysis_stats["avg_analysis_time_ms"] == 150.0  # (100 + 200) / 2
        assert analyzer.analysis_stats["quality_score_average"] == 0.7  # (0.8 + 0.6) / 2

    @pytest.mark.asyncio
    async def test_get_analysis_statistics(self):
        """Test getting analysis statistics."""
        analyzer = WorkflowAnalyzer()

        # Add some test data
        analyzer.analysis_stats["total_analyses"] = 5
        analyzer.analysis_stats["patterns_identified"] = 10
        analyzer.pattern_library["test_pattern"] = Mock()
        analyzer.analysis_history["test_analysis"] = Mock()

        stats = await analyzer.get_analysis_statistics()

        assert stats["total_analyses_performed"] == 5
        assert stats["patterns_identified"] == 10
        assert stats["pattern_library_size"] == 3  # 2 default + 1 test
        assert stats["analysis_history_size"] == 1
        assert "last_updated" in stats

    def test_find_similar_components(self):
        """Test finding similar components."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="file_comp1",
                component_type="action",
                name="File Reader Alpha",
                description="Reads files",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="file_comp2",
                component_type="action",
                name="File Reader Beta",  # Similar name
                description="Also reads files",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="calc_comp",
                component_type="action",
                name="Calculator",
                description="Does calculations",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        similar = analyzer._find_similar_components(components)

        # Should find the two file reader components as similar
        assert len(similar) == 2
        assert any(comp.component_id == "file_comp1" for comp in similar)
        assert any(comp.component_id == "file_comp2" for comp in similar)

    def test_analyze_cross_tool_dependencies(self):
        """Test cross-tool dependency analysis."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="tool1_comp",
                component_type="action",
                name="Tool 1 Component",
                description="Uses tool 1",
                parameters={"tool": "tool1"},
                dependencies=["tool2_comp"],  # Depends on tool 2
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="tool2_comp",
                component_type="action",
                name="Tool 2 Component",
                description="Uses tool 2",
                parameters={"tool": "tool2"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        dependencies = analyzer._analyze_cross_tool_dependencies(components)

        assert "tool1" in dependencies
        assert "tool2" in dependencies["tool1"]

    @pytest.mark.asyncio
    async def test_assess_resource_requirements(self):
        """Test resource requirements assessment."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="file_comp",
                component_type="action",
                name="File Component",
                description="Works with files",
                parameters={"file_path": "/tmp/test.txt"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.3,
            ),
            WorkflowComponent(
                component_id="api_comp",
                component_type="action",
                name="API Component",
                description="Calls API",
                parameters={"url": "https://api.example.com"},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.4,
            ),
        ]

        requirements = await analyzer._assess_resource_requirements(components)

        assert "cpu_percentage" in requirements
        assert "memory_mb" in requirements
        assert "network_mb" in requirements
        assert "storage_mb" in requirements
        assert requirements["parallel_execution_capable"] is True  # No dependencies
        assert requirements["external_dependencies"] == 1  # One URL

    def test_assess_reliability(self):
        """Test reliability assessment."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="reliable_comp",
                component_type="action",
                name="Reliable Component",
                description="High reliability",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.95,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="unreliable_comp",
                component_type="action",
                name="Unreliable Component",
                description="Lower reliability",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.7,
                complexity_score=0.2,
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=2,
            unique_component_types=1,
            dependency_depth=1,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        reliability = analyzer._assess_reliability(components, metrics)

        assert "overall" in reliability
        assert "component_average" in reliability
        assert "structure_penalty" in reliability
        assert reliability["component_average"] == 0.825  # (0.95 + 0.7) / 2

    def test_calculate_maintainability(self):
        """Test maintainability calculation."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="maintainable_comp",
                component_type="action",
                name="Well Documented Component",
                description="This is a well-documented component with good maintainability characteristics",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=2,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        # Mock calculate_workflow_complexity_score
        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.3,
        ):
            maintainability = analyzer._calculate_maintainability(components, metrics)

        assert 0.0 <= maintainability <= 1.0
        assert maintainability > 0.5  # Should be reasonably maintainable


class TestWorkflowAnalyzerMainAnalysisFlow:
    """Comprehensive tests for the main analyze_workflow method."""

    @pytest.mark.asyncio
    async def test_analyze_workflow_success(self):
        """Test successful workflow analysis."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "workflow_id": "test_workflow_001",
            "components": [
                {
                    "component_id": "comp_001",
                    "component_type": "action",
                    "name": "Test Action",
                    "description": "Test action component",
                    "parameters": {"param1": "value1"},
                    "dependencies": [],
                    "execution_time_ms": 500,
                    "reliability_score": 0.9,
                    "complexity_score": 0.3,
                }
            ],
        }

        # Mock all the complex analysis methods to control the test
        mock_metrics_result = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        with patch.object(
            analyzer, "_collect_analysis_metrics", return_value=mock_metrics_result
        ) as mock_metrics, patch.object(
            analyzer, "_assess_workflow_quality", return_value=0.8
        ) as mock_quality, patch.object(
            analyzer, "_analyze_complexity", return_value={}
        ) as mock_complexity, patch.object(
            analyzer, "_predict_performance", return_value={}
        ) as mock_performance, patch.object(
            analyzer, "_identify_patterns", return_value=[]
        ) as mock_patterns, patch.object(
            analyzer, "_detect_anti_patterns", return_value=[]
        ) as mock_anti_patterns, patch.object(
            analyzer, "_generate_optimizations", return_value=[]
        ) as mock_optimizations, patch.object(
            analyzer, "_generate_improvement_suggestions", return_value=[]
        ) as mock_suggestions, patch.object(
            analyzer, "_generate_alternative_designs", return_value=[]
        ) as mock_alternatives:

            result = await analyzer.analyze_workflow(workflow_data)

            assert result.is_right()
            analysis_result = result.get_right()
            assert analysis_result.workflow_id == "test_workflow_001"
            assert analysis_result.quality_score == 0.8
            assert analysis_result.analysis_depth == IntelligenceLevel.ML_ENHANCED

            # Verify all analysis methods were called
            mock_metrics.assert_called_once()
            mock_quality.assert_called_once()
            mock_complexity.assert_called_once()
            mock_performance.assert_called_once()
            mock_patterns.assert_called_once()
            mock_anti_patterns.assert_called_once()
            mock_optimizations.assert_called_once()
            mock_suggestions.assert_called_once()
            mock_alternatives.assert_called_once()

    @pytest.mark.asyncio
    async def test_analyze_workflow_no_components(self):
        """Test workflow analysis with no valid components."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {"workflow_id": "empty_workflow"}

        result = await analyzer.analyze_workflow(workflow_data)

        assert result.is_left()
        error = result.get_left()
        assert isinstance(error, ValidationError)
        assert "no valid components found" in str(error)

    @pytest.mark.asyncio
    async def test_analyze_workflow_with_custom_optimization_goals(self):
        """Test workflow analysis with custom optimization goals."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "components": [
                {
                    "component_id": "comp_001",
                    "component_type": "action",
                    "name": "Test Action",
                    "description": "Test action",
                    "parameters": {},
                    "dependencies": [],
                    "execution_time_ms": 500,
                    "reliability_score": 0.9,
                    "complexity_score": 0.3,
                }
            ]
        }

        optimization_goals = [OptimizationGoal.PERFORMANCE, OptimizationGoal.RELIABILITY]

        # Mock methods to focus on testing the optimization goals flow
        mock_metrics_result = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        with patch.object(analyzer, "_collect_analysis_metrics", return_value=mock_metrics_result), patch.object(
            analyzer, "_assess_workflow_quality", return_value=0.8
        ), patch.object(analyzer, "_analyze_complexity", return_value={}), patch.object(
            analyzer, "_predict_performance", return_value={}
        ), patch.object(
            analyzer, "_identify_patterns", return_value=[]
        ), patch.object(
            analyzer, "_detect_anti_patterns", return_value=[]
        ), patch.object(
            analyzer, "_generate_optimizations", return_value=[]
        ) as mock_optimizations, patch.object(
            analyzer, "_generate_improvement_suggestions", return_value=[]
        ), patch.object(
            analyzer, "_generate_alternative_designs", return_value=[]
        ):

            result = await analyzer.analyze_workflow(
                workflow_data, optimization_goals=optimization_goals
            )

            assert result.is_right()
            # Verify optimization goals were passed correctly
            mock_optimizations.assert_called_once()
            args = mock_optimizations.call_args[0]
            passed_goals = mock_optimizations.call_args[1]["optimization_goals"]
            assert passed_goals == optimization_goals

    @pytest.mark.asyncio
    async def test_analyze_workflow_exception_handling(self):
        """Test workflow analysis exception handling."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "components": [
                {
                    "component_id": "comp_001",
                    "component_type": "action",
                    "name": "Test Action",
                    "description": "Test action",
                }
            ]
        }

        # Mock a method to raise an exception
        with patch.object(
            analyzer,
            "_collect_analysis_metrics",
            side_effect=Exception("Test exception"),
        ):
            result = await analyzer.analyze_workflow(workflow_data)

            assert result.is_left()
            error = result.get_left()
            assert isinstance(error, ValidationError)
            assert "Test exception" in str(error)

    @pytest.mark.asyncio
    async def test_analyze_workflow_stores_in_history(self):
        """Test that workflow analysis results are stored in history."""
        analyzer = WorkflowAnalyzer()
        workflow_data = {
            "components": [
                {
                    "component_id": "comp_001",
                    "component_type": "action",
                    "name": "Test Action",
                    "description": "Test action",
                }
            ]
        }

        # Mock all analysis methods
        mock_metrics_result = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        with patch.object(analyzer, "_collect_analysis_metrics", return_value=mock_metrics_result), patch.object(
            analyzer, "_assess_workflow_quality", return_value=0.8
        ), patch.object(analyzer, "_analyze_complexity", return_value={}), patch.object(
            analyzer, "_predict_performance", return_value={}
        ), patch.object(
            analyzer, "_identify_patterns", return_value=[]
        ), patch.object(
            analyzer, "_detect_anti_patterns", return_value=[]
        ), patch.object(
            analyzer, "_generate_optimizations", return_value=[]
        ), patch.object(
            analyzer, "_generate_improvement_suggestions", return_value=[]
        ), patch.object(
            analyzer, "_generate_alternative_designs", return_value=[]
        ):

            initial_history_size = len(analyzer.analysis_history)
            result = await analyzer.analyze_workflow(workflow_data)

            assert result.is_right()
            # Verify result was stored in history
            assert len(analyzer.analysis_history) == initial_history_size + 1


class TestWorkflowAnalyzerOptimizationGeneration:
    """Comprehensive tests for optimization generation methods."""

    @pytest.mark.asyncio
    async def test_generate_performance_optimizations(self):
        """Test performance optimization generation."""
        analyzer = WorkflowAnalyzer()
        # Create many independent actions for parallelization opportunity
        components = []
        for i in range(5):
            components.append(
                WorkflowComponent(
                    component_id=f"action_{i}",
                    component_type="action",
                    name=f"Independent Action {i}",
                    description="Independent action",
                    parameters={},
                    dependencies=[],  # No dependencies = independent
                    estimated_execution_time=timedelta(milliseconds=100),
                    reliability_score=0.9,
                    complexity_score=0.2,
                )
            )

        metrics = Mock()

        with patch(
            "src.intelligence.workflow_analyzer.create_optimization_id"
        ) as mock_opt_id, patch(
            "src.intelligence.workflow_analyzer.create_recommendation_id"
        ) as mock_rec_id:
            mock_opt_id.return_value = "opt_123"
            mock_rec_id.return_value = "rec_456"

            optimizations = await analyzer._generate_performance_optimizations(
                components, metrics
            )

            assert len(optimizations) > 0
            optimization = optimizations[0]
            assert "Parallelize" in optimization.title
            assert optimization.optimization_goals == [OptimizationGoal.PERFORMANCE]

    @pytest.mark.asyncio
    async def test_generate_efficiency_optimizations(self):
        """Test efficiency optimization generation."""
        analyzer = WorkflowAnalyzer()
        # Create similar components for consolidation
        components = [
            WorkflowComponent(
                component_id="file_comp1",
                component_type="action",
                name="File Reader Alpha",
                description="Reads files",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
            WorkflowComponent(
                component_id="file_comp2",
                component_type="action",
                name="File Reader Beta",  # Similar name
                description="Also reads files",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.2,
            ),
        ]

        metrics = Mock()

        with patch(
            "src.intelligence.workflow_analyzer.create_optimization_id"
        ) as mock_opt_id, patch(
            "src.intelligence.workflow_analyzer.create_recommendation_id"
        ) as mock_rec_id:
            mock_opt_id.return_value = "opt_123"
            mock_rec_id.return_value = "rec_456"

            optimizations = await analyzer._generate_efficiency_optimizations(
                components, metrics
            )

            assert len(optimizations) > 0
            optimization = optimizations[0]
            assert "Consolidate" in optimization.title
            assert optimization.optimization_goals == [OptimizationGoal.EFFICIENCY]

    @pytest.mark.asyncio
    async def test_generate_reliability_optimizations(self):
        """Test reliability optimization generation."""
        analyzer = WorkflowAnalyzer()
        components = []

        # Create metrics indicating missing error handling
        metrics = AnalysisMetrics(
            total_components=5,
            unique_component_types=2,
            dependency_depth=1,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=["No error handling detected"],
        )

        with patch(
            "src.intelligence.workflow_analyzer.create_optimization_id"
        ) as mock_opt_id, patch(
            "src.intelligence.workflow_analyzer.create_recommendation_id"
        ) as mock_rec_id:
            mock_opt_id.return_value = "opt_123"
            mock_rec_id.return_value = "rec_456"

            optimizations = await analyzer._generate_reliability_optimizations(
                components, metrics
            )

            assert len(optimizations) > 0
            optimization = optimizations[0]
            assert "Error Handling" in optimization.title
            assert optimization.optimization_goals == [OptimizationGoal.RELIABILITY]

    @pytest.mark.asyncio
    async def test_generate_optimizations_multiple_goals(self):
        """Test optimization generation with multiple goals."""
        analyzer = WorkflowAnalyzer()
        components = []
        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=0,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=["No error handling detected"],
        )

        optimization_goals = [
            OptimizationGoal.PERFORMANCE,
            OptimizationGoal.EFFICIENCY,
            OptimizationGoal.RELIABILITY,
        ]

        with patch.object(
            analyzer, "_generate_performance_optimizations", return_value=[]
        ) as mock_perf, patch.object(
            analyzer, "_generate_efficiency_optimizations", return_value=[]
        ) as mock_eff, patch.object(
            analyzer, "_generate_reliability_optimizations", return_value=[]
        ) as mock_rel:

            await analyzer._generate_optimizations(components, metrics, optimization_goals)

            # Verify all optimization methods were called
            mock_perf.assert_called_once_with(components, metrics)
            mock_eff.assert_called_once_with(components, metrics)
            mock_rel.assert_called_once_with(components, metrics)


class TestWorkflowAnalyzerAlternativeDesigns:
    """Comprehensive tests for alternative design generation."""

    @pytest.mark.asyncio
    async def test_generate_alternative_designs_simplified(self):
        """Test generation of simplified alternative design."""
        analyzer = WorkflowAnalyzer()
        # Mix of simple and complex components
        components = [
            WorkflowComponent(
                component_id="simple_comp",
                component_type="action",
                name="Simple Component",
                description="Simple operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.3,  # Simple
            ),
            WorkflowComponent(
                component_id="complex_comp",
                component_type="action",
                name="Complex Component",
                description="Complex operation",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.8,  # Complex - will be filtered out
            ),
        ]

        optimization_goals = [OptimizationGoal.EFFICIENCY]

        alternatives = await analyzer._generate_alternative_designs(
            components, optimization_goals
        )

        assert len(alternatives) > 0
        simplified = next(
            alt for alt in alternatives if "Simplified" in alt["name"]
        )
        assert simplified["component_count"] == 1  # Only simple component


    @pytest.mark.asyncio
    async def test_generate_alternative_designs_performance_optimized(self):
        """Test generation of performance-optimized alternative design."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Component 1",
                description="Action 1",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.3,
            ),
        ]

        optimization_goals = [OptimizationGoal.PERFORMANCE]

        alternatives = await analyzer._generate_alternative_designs(
            components, optimization_goals
        )

        assert len(alternatives) > 0
        performance_opt = next(
            alt for alt in alternatives if "Performance-Optimized" in alt["name"]
        )
        assert "execution_time" in performance_opt["estimated_improvement"]

    @pytest.mark.asyncio
    async def test_generate_alternative_designs_reliability_focused(self):
        """Test generation of reliability-focused alternative design."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="comp1",
                component_type="action",
                name="Component 1",
                description="Action 1",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.9,
                complexity_score=0.3,
            ),
        ]

        optimization_goals = [OptimizationGoal.RELIABILITY]

        alternatives = await analyzer._generate_alternative_designs(
            components, optimization_goals
        )

        assert len(alternatives) > 0
        reliability_focused = next(
            alt for alt in alternatives if "High-Reliability" in alt["name"]
        )
        assert reliability_focused["component_count"] == 3  # Original + 2 error handling


class TestWorkflowAnalyzerImprovementSuggestions:
    """Comprehensive tests for improvement suggestion generation."""

    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_comprehensive(self):
        """Test comprehensive improvement suggestion generation."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="unreliable_comp",
                component_type="action",
                name="Unreliable Component",
                description="Low reliability component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.6,  # Low reliability
                complexity_score=0.9,  # High complexity
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=5,
            cyclic_dependencies=True,
            resource_conflicts=["File conflict detected"],
            performance_bottlenecks=["Slow operation detected"],
            reliability_concerns=["Low reliability component"],
        )

        # Create anti-pattern for testing
        anti_pattern = WorkflowPattern(
            pattern_id="anti_pattern_001",
            pattern_type=PatternType.ANTI_PATTERN,
            name="Test Anti-Pattern",
            description="Test anti-pattern",
            components=[],
            usage_frequency=0.1,
            effectiveness_score=0.3,
            complexity_reduction=-0.2,
            reusability_score=0.2,
            detected_in_workflows=[],
            template_generated=False,
            confidence_score=0.8,
        )

        patterns = []
        anti_patterns = [anti_pattern]

        # Mock calculate_workflow_complexity_score to return high complexity
        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.9,
        ):
            suggestions = await analyzer._generate_improvement_suggestions(
                components, metrics, patterns, anti_patterns
            )

        # Should have multiple suggestions based on the issues
        assert len(suggestions) > 0
        suggestion_text = " ".join(suggestions).lower()

        # Check for expected suggestion types
        assert any(
            keyword in suggestion_text
            for keyword in [
                "conflict",
                "bottleneck",
                "cyclic",
                "anti-pattern",
                "complex",
                "reliability",
            ]
        )

    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_no_issues(self):
        """Test improvement suggestions when no issues are found."""
        analyzer = WorkflowAnalyzer()
        components = [
            WorkflowComponent(
                component_id="perfect_comp",
                component_type="action",
                name="Perfect Component",
                description="Well-designed component",
                parameters={},
                dependencies=[],
                estimated_execution_time=timedelta(milliseconds=100),
                reliability_score=0.95,  # High reliability
                complexity_score=0.2,  # Low complexity
            ),
        ]

        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=1,
            cyclic_dependencies=False,
            resource_conflicts=[],
            performance_bottlenecks=[],
            reliability_concerns=[],
        )

        patterns = []
        anti_patterns = []

        # Mock calculate_workflow_complexity_score to return low complexity
        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.2,
        ):
            suggestions = await analyzer._generate_improvement_suggestions(
                components, metrics, patterns, anti_patterns
            )

        # Should have default suggestion when no issues found
        assert len(suggestions) == 1
        assert "monitoring" in suggestions[0].lower() or "logging" in suggestions[0].lower()

    @pytest.mark.asyncio
    async def test_generate_improvement_suggestions_limit_to_five(self):
        """Test that improvement suggestions are limited to top 5."""
        analyzer = WorkflowAnalyzer()
        components = []

        # Create metrics with many issues to generate many suggestions
        metrics = AnalysisMetrics(
            total_components=1,
            unique_component_types=1,
            dependency_depth=10,
            cyclic_dependencies=True,
            resource_conflicts=["conflict1", "conflict2", "conflict3"],
            performance_bottlenecks=["bottleneck1", "bottleneck2"],
            reliability_concerns=["concern1"],
        )

        # Create multiple anti-patterns
        anti_patterns = []
        for i in range(10):
            anti_pattern = WorkflowPattern(
                pattern_id=f"anti_pattern_{i}",
                pattern_type=PatternType.ANTI_PATTERN,
                name=f"Anti-Pattern {i}",
                description=f"Anti-pattern {i}",
                components=[],
                usage_frequency=0.1,
                effectiveness_score=0.3,
                complexity_reduction=-0.2,
                reusability_score=0.2,
                detected_in_workflows=[],
                template_generated=False,
                confidence_score=0.8,
            )
            anti_patterns.append(anti_pattern)

        patterns = []

        with patch(
            "src.intelligence.workflow_analyzer.calculate_workflow_complexity_score",
            return_value=0.9,
        ):
            suggestions = await analyzer._generate_improvement_suggestions(
                components, metrics, patterns, anti_patterns
            )

        # Should be limited to 5 suggestions
        assert len(suggestions) <= 5
