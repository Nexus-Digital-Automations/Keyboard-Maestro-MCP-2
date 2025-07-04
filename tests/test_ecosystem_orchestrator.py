"""
Comprehensive test suite for ecosystem orchestrator functionality.

Tests the complete ecosystem orchestration system including workflow execution,
performance optimization, strategic planning, and tool coordination.

Security: Enterprise-grade test validation with comprehensive security coverage.
Performance: Test execution optimized for comprehensive coverage.
Type Safety: Complete integration with ecosystem architecture testing.
"""

import pytest
import asyncio
from typing import Dict, List, Any, Set
from datetime import datetime, timedelta, UTC
from unittest.mock import Mock, AsyncMock, patch
from hypothesis import given, strategies as st, settings

from src.orchestration.ecosystem_orchestrator import (
    EcosystemOrchestrator, OrchestrationError
)
from src.orchestration.performance_monitor import EcosystemPerformanceMonitor
from src.orchestration.workflow_engine import MasterWorkflowEngine
from src.orchestration.strategic_planner import EcosystemStrategicPlanner
from src.orchestration.ecosystem_architecture import (
    ToolDescriptor, ToolRegistry, ToolCategory, ExecutionMode, OptimizationTarget,
    EcosystemWorkflow, WorkflowStep, SystemPerformanceMetrics, OrchestrationResult,
    ResourceType, SecurityLevel, create_workflow_id, create_step_id, create_orchestration_id,
    calculate_workflow_complexity, estimate_workflow_duration, validate_workflow_security
)
from src.core.either import Either
from src.core.contracts import require, ensure
from src.core.errors import ValidationError


class TestToolDescriptor:
    """Test tool descriptor functionality and validation."""
    
    @pytest.fixture
    def sample_tool(self):
        """Create sample tool descriptor for testing."""
        return ToolDescriptor(
            tool_id="km_test_tool",
            tool_name="Test Tool",
            category=ToolCategory.FOUNDATION,
            capabilities={"test_capability", "validation"},
            dependencies=["km_dependency"],
            resource_requirements={"cpu": 0.2, "memory": 0.1},
            performance_characteristics={"response_time": 1.0, "reliability": 0.95},
            integration_points=["km_other_tool"],
            security_level=SecurityLevel.STANDARD,
            enterprise_ready=True,
            ai_enhanced=False
        )
    
    def test_tool_descriptor_creation(self, sample_tool):
        """Test tool descriptor creation with valid parameters."""
        assert sample_tool.tool_id == "km_test_tool"
        assert sample_tool.category == ToolCategory.FOUNDATION
        assert len(sample_tool.capabilities) == 2
        assert sample_tool.enterprise_ready is True
        assert sample_tool.ai_enhanced is False
    
    def test_tool_compatibility_check(self, sample_tool):
        """Test tool compatibility validation."""
        compatible_tool = ToolDescriptor(
            tool_id="km_compatible",
            tool_name="Compatible Tool",
            category=ToolCategory.INTELLIGENCE,
            capabilities={"different_capability"},
            dependencies=[],
            resource_requirements={"cpu": 0.3, "memory": 0.2},
            performance_characteristics={"response_time": 0.5, "reliability": 0.90},
            integration_points=[],
            security_level=SecurityLevel.STANDARD
        )
        
        # Compatible tools should return True
        assert sample_tool.is_compatible_with(compatible_tool)
        
        # Tool with conflicting resources should return False
        conflicting_tool = ToolDescriptor(
            tool_id="km_conflicting",
            tool_name="Conflicting Tool",
            category=ToolCategory.INTELLIGENCE,
            capabilities={"conflicting_capability"},
            dependencies=[],
            resource_requirements={"cpu": 0.9, "memory": 0.8},  # High resource usage
            performance_characteristics={"response_time": 2.0, "reliability": 0.85},
            integration_points=[],
            security_level=SecurityLevel.STANDARD
        )
        
        assert not sample_tool.is_compatible_with(conflicting_tool)
    
    def test_tool_synergy_calculation(self, sample_tool):
        """Test synergy score calculation between tools."""
        synergistic_tool = ToolDescriptor(
            tool_id="km_synergistic",
            tool_name="Synergistic Tool",
            category=ToolCategory.INTELLIGENCE,  # Different category for synergy
            capabilities={"test_capability", "shared_capability"},  # Shared capability
            dependencies=[],
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            performance_characteristics={"response_time": 0.8, "reliability": 0.92},
            integration_points=["km_other_tool"],  # Shared integration point
            security_level=SecurityLevel.STANDARD
        )
        
        synergy_score = sample_tool.get_synergy_score(synergistic_tool)
        assert 0.0 <= synergy_score <= 1.0
        assert synergy_score > 0.3  # Should have significant synergy
    
    @given(
        tool_id=st.text(min_size=1, max_size=50),
        capabilities=st.sets(st.text(min_size=1, max_size=20), min_size=1, max_size=10)
    )
    @settings(max_examples=50)
    def test_tool_descriptor_property_validation(self, tool_id, capabilities):
        """Property-based test for tool descriptor validation."""
        try:
            tool = ToolDescriptor(
                tool_id=tool_id,
                tool_name=f"Test {tool_id}",
                category=ToolCategory.FOUNDATION,
                capabilities=capabilities,
                dependencies=[],
                resource_requirements={"cpu": 0.1, "memory": 0.1},
                performance_characteristics={"response_time": 1.0, "reliability": 0.9},
                integration_points=[],
                security_level=SecurityLevel.STANDARD
            )
            
            # Validate required properties
            assert len(tool.tool_id) > 0
            assert len(tool.tool_name) > 0
            assert len(tool.capabilities) > 0
            
        except Exception as e:
            # Contract violations should raise ValidationError
            assert isinstance(e, (ValidationError, ValueError))


class TestToolRegistry:
    """Test tool registry functionality and management."""
    
    @pytest.fixture
    def tool_registry(self):
        """Create tool registry with sample tools."""
        registry = ToolRegistry()
        
        # Add foundation tool
        foundation_tool = ToolDescriptor(
            tool_id="km_foundation",
            tool_name="Foundation Tool",
            category=ToolCategory.FOUNDATION,
            capabilities={"macro_creation", "validation"},
            dependencies=[],
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            performance_characteristics={"response_time": 0.5, "reliability": 0.95},
            integration_points=[],
            security_level=SecurityLevel.STANDARD,
            enterprise_ready=True
        )
        registry.register_tool(foundation_tool)
        
        # Add intelligence tool
        intelligence_tool = ToolDescriptor(
            tool_id="km_intelligence",
            tool_name="Intelligence Tool",
            category=ToolCategory.INTELLIGENCE,
            capabilities={"ai_processing", "pattern_recognition"},
            dependencies=["km_foundation"],
            resource_requirements={"cpu": 0.5, "memory": 0.3},
            performance_characteristics={"response_time": 2.0, "reliability": 0.88},
            integration_points=["km_foundation"],
            security_level=SecurityLevel.HIGH,
            enterprise_ready=True,
            ai_enhanced=True
        )
        registry.register_tool(intelligence_tool)
        
        return registry
    
    def test_tool_registration(self, tool_registry):
        """Test tool registration and indexing."""
        assert len(tool_registry.tools) == 2
        assert "km_foundation" in tool_registry.tools
        assert "km_intelligence" in tool_registry.tools
        
        # Test capability indexing
        assert "macro_creation" in tool_registry.capability_index
        assert "ai_processing" in tool_registry.capability_index
        
        # Test category indexing
        assert ToolCategory.FOUNDATION in tool_registry.category_index
        assert ToolCategory.INTELLIGENCE in tool_registry.category_index
    
    def test_find_tools_by_capability(self, tool_registry):
        """Test finding tools by capability."""
        creation_tools = tool_registry.find_tools_by_capability("macro_creation")
        assert len(creation_tools) == 1
        assert creation_tools[0].tool_id == "km_foundation"
        
        ai_tools = tool_registry.find_tools_by_capability("ai_processing")
        assert len(ai_tools) == 1
        assert ai_tools[0].tool_id == "km_intelligence"
    
    def test_find_tools_by_category(self, tool_registry):
        """Test finding tools by category."""
        foundation_tools = tool_registry.find_tools_by_category(ToolCategory.FOUNDATION)
        assert len(foundation_tools) == 1
        assert foundation_tools[0].tool_id == "km_foundation"
        
        intelligence_tools = tool_registry.find_tools_by_category(ToolCategory.INTELLIGENCE)
        assert len(intelligence_tools) == 1
        assert intelligence_tools[0].tool_id == "km_intelligence"
    
    def test_tool_synergies(self, tool_registry):
        """Test tool synergy identification."""
        synergies = tool_registry.get_tool_synergies("km_foundation")
        assert isinstance(synergies, list)
        
        # Should find synergy with intelligence tool
        if synergies:
            synergy_ids = [s[0] for s in synergies]
            synergy_scores = [s[1] for s in synergies]
            assert all(0.0 <= score <= 1.0 for score in synergy_scores)


class TestWorkflowStep:
    """Test workflow step creation and validation."""
    
    def test_workflow_step_creation(self):
        """Test workflow step creation with valid parameters."""
        step = WorkflowStep(
            step_id=create_step_id(),
            tool_id="km_test_tool",
            parameters={"input": "test_value"},
            preconditions=["condition1"],
            postconditions=["result1"],
            timeout=300,
            retry_count=3,
            parallel_group="group1"
        )
        
        assert len(step.step_id) > 0
        assert step.tool_id == "km_test_tool"
        assert step.parameters["input"] == "test_value"
        assert step.timeout == 300
        assert step.retry_count == 3
        assert step.parallel_group == "group1"
    
    @given(
        timeout=st.integers(min_value=1, max_value=3600),
        retry_count=st.integers(min_value=0, max_value=10)
    )
    @settings(max_examples=50)
    def test_workflow_step_property_validation(self, timeout, retry_count):
        """Property-based test for workflow step validation."""
        step = WorkflowStep(
            step_id=create_step_id(),
            tool_id="km_property_test",
            parameters={},
            timeout=timeout,
            retry_count=retry_count
        )
        
        assert step.timeout > 0
        assert step.retry_count >= 0


class TestEcosystemWorkflow:
    """Test ecosystem workflow creation and management."""
    
    @pytest.fixture
    def sample_workflow(self):
        """Create sample workflow for testing."""
        steps = [
            WorkflowStep(
                step_id=create_step_id(),
                tool_id="km_foundation",
                parameters={"action": "create"},
                timeout=120,
                retry_count=2
            ),
            WorkflowStep(
                step_id=create_step_id(),
                tool_id="km_intelligence",
                parameters={"mode": "analyze"},
                timeout=300,
                retry_count=3,
                parallel_group="analysis"
            )
        ]
        
        return EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Test Workflow",
            description="Sample workflow for testing",
            steps=steps,
            execution_mode=ExecutionMode.PARALLEL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=60.0,
            resource_requirements={"cpu": 0.6, "memory": 0.4}
        )
    
    def test_workflow_creation(self, sample_workflow):
        """Test workflow creation with valid parameters."""
        assert len(sample_workflow.workflow_id) > 0
        assert sample_workflow.name == "Test Workflow"
        assert len(sample_workflow.steps) == 2
        assert sample_workflow.execution_mode == ExecutionMode.PARALLEL
        assert sample_workflow.optimization_target == OptimizationTarget.PERFORMANCE
        assert sample_workflow.expected_duration == 60.0
    
    def test_workflow_tool_dependencies(self, sample_workflow):
        """Test workflow tool dependency analysis."""
        dependencies = sample_workflow.get_tool_dependencies()
        assert isinstance(dependencies, dict)
        assert len(dependencies) == 2
        
        # Both tools should have empty dependency lists in this simple workflow
        for tool_deps in dependencies.values():
            assert isinstance(tool_deps, list)
    
    def test_workflow_parallel_groups(self, sample_workflow):
        """Test workflow parallel group identification."""
        parallel_groups = sample_workflow.get_parallel_groups()
        assert isinstance(parallel_groups, dict)
        assert "analysis" in parallel_groups
        assert len(parallel_groups["analysis"]) == 1


class TestSystemPerformanceMetrics:
    """Test system performance metrics and health scoring."""
    
    def test_performance_metrics_creation(self):
        """Test performance metrics creation and validation."""
        metrics = SystemPerformanceMetrics(
            timestamp=datetime.now(UTC),
            total_tools_active=48,
            resource_utilization={"cpu": 0.6, "memory": 0.4, "network": 0.2},
            average_response_time=1.2,
            success_rate=0.94,
            error_rate=0.02,
            throughput=150.0,
            bottlenecks=["cpu_intensive_operations"],
            optimization_opportunities=["cache_optimization", "parallel_execution"]
        )
        
        assert metrics.total_tools_active == 48
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.average_response_time >= 0.0
        assert metrics.throughput >= 0.0
    
    def test_health_score_calculation(self):
        """Test health score calculation algorithm."""
        # High performance metrics
        high_metrics = SystemPerformanceMetrics(
            timestamp=datetime.now(UTC),
            total_tools_active=48,
            resource_utilization={"cpu": 0.3, "memory": 0.2, "network": 0.1},
            average_response_time=0.5,
            success_rate=0.98,
            error_rate=0.01,
            throughput=200.0,
            bottlenecks=[],
            optimization_opportunities=[]
        )
        
        high_score = high_metrics.get_health_score()
        assert 0.8 <= high_score <= 1.0
        
        # Low performance metrics
        low_metrics = SystemPerformanceMetrics(
            timestamp=datetime.now(UTC),
            total_tools_active=48,
            resource_utilization={"cpu": 0.9, "memory": 0.8, "network": 0.7},
            average_response_time=5.0,
            success_rate=0.7,
            error_rate=0.2,
            throughput=50.0,
            bottlenecks=["cpu_overload", "memory_pressure", "network_saturation"],
            optimization_opportunities=["urgent_optimization_needed"]
        )
        
        low_score = low_metrics.get_health_score()
        assert 0.0 <= low_score <= 0.5
    
    @given(
        success_rate=st.floats(min_value=0.0, max_value=1.0),
        error_rate=st.floats(min_value=0.0, max_value=1.0),
        response_time=st.floats(min_value=0.0, max_value=10.0),
        throughput=st.floats(min_value=0.0, max_value=1000.0)
    )
    @settings(max_examples=100)
    def test_health_score_properties(self, success_rate, error_rate, response_time, throughput):
        """Property-based test for health score calculation."""
        metrics = SystemPerformanceMetrics(
            timestamp=datetime.now(UTC),
            total_tools_active=10,
            resource_utilization={"cpu": 0.5},
            average_response_time=response_time,
            success_rate=success_rate,
            error_rate=error_rate,
            throughput=throughput,
            bottlenecks=[],
            optimization_opportunities=[]
        )
        
        health_score = metrics.get_health_score()
        assert 0.0 <= health_score <= 1.0


class TestEcosystemPerformanceMonitor:
    """Test performance monitoring system."""
    
    @pytest.fixture
    def performance_monitor(self):
        """Create performance monitor for testing."""
        monitor = EcosystemPerformanceMonitor()
        # Simulate some tool performance data
        from src.orchestration.performance_monitor import ToolPerformanceMetrics
        from datetime import datetime, UTC
        
        from src.orchestration.ecosystem_architecture import ToolCategory
        
        monitor.tool_performance["test_tool_1"] = ToolPerformanceMetrics(
            tool_id="test_tool_1",
            tool_name="test_tool_1",
            category=ToolCategory.FOUNDATION,
            execution_count=10,
            total_execution_time=5.0,
            average_response_time=0.5,
            success_rate=0.95,
            error_rate=0.05,
            resource_efficiency=0.8,
            last_execution=datetime.now(UTC)
        )
        monitor.tool_performance["test_tool_2"] = ToolPerformanceMetrics(
            tool_id="test_tool_2",
            tool_name="test_tool_2", 
            category=ToolCategory.DATA_MANAGEMENT,
            execution_count=5,
            total_execution_time=6.0,
            average_response_time=1.2,
            success_rate=0.98,
            error_rate=0.02,
            resource_efficiency=0.9,
            last_execution=datetime.now(UTC)
        )
        return monitor
    
    @pytest.mark.asyncio
    async def test_get_current_metrics(self, performance_monitor):
        """Test current metrics retrieval."""
        metrics = await performance_monitor.get_current_metrics()
        
        assert isinstance(metrics, SystemPerformanceMetrics)
        assert metrics.total_tools_active > 0
        assert isinstance(metrics.resource_utilization, dict)
        assert metrics.average_response_time >= 0.0
        assert 0.0 <= metrics.success_rate <= 1.0
        assert 0.0 <= metrics.error_rate <= 1.0
        assert metrics.throughput >= 0.0
    
    @pytest.mark.asyncio
    async def test_bottleneck_detection(self, performance_monitor):
        """Test bottleneck detection algorithm."""
        # Generate some metrics history
        for _ in range(5):
            await performance_monitor.get_current_metrics()
            await asyncio.sleep(0.01)  # Small delay to simulate time passage
        
        bottlenecks = await performance_monitor.detect_bottlenecks()
        assert isinstance(bottlenecks, list)
        assert all(isinstance(b, str) for b in bottlenecks)


class TestMasterWorkflowEngine:
    """Test workflow execution engine."""
    
    @pytest.fixture
    def workflow_engine(self):
        """Create workflow engine with sample tool registry."""
        registry = ToolRegistry()
        
        # Register test tools
        test_tool = ToolDescriptor(
            tool_id="km_test_execution",
            tool_name="Test Execution Tool",
            category=ToolCategory.FOUNDATION,
            capabilities={"test_execution"},
            dependencies=[],
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            performance_characteristics={"response_time": 0.5, "reliability": 0.95},
            integration_points=[],
            security_level=SecurityLevel.STANDARD
        )
        registry.register_tool(test_tool)
        
        return MasterWorkflowEngine(registry)
    
    @pytest.mark.asyncio
    async def test_workflow_validation(self, workflow_engine):
        """Test workflow validation before execution."""
        # Valid workflow
        valid_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Valid Test Workflow",
            description="Test workflow validation",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_test_execution",
                    parameters={"test": "value"}
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=30.0,
            resource_requirements={}
        )
        
        result = workflow_engine._validate_workflow(valid_workflow)
        assert result.is_right()
        
        # Invalid workflow with non-existent tool
        invalid_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Invalid Test Workflow",
            description="Test workflow validation",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_nonexistent_tool",
                    parameters={"test": "value"}
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=30.0,
            resource_requirements={}
        )
        
        result = workflow_engine._validate_workflow(invalid_workflow)
        assert result.is_left()
        assert isinstance(result.get_left(), OrchestrationError)
    
    @pytest.mark.asyncio
    async def test_execution_plan_optimization(self, workflow_engine):
        """Test execution plan optimization."""
        workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Optimization Test Workflow",
            description="Test execution plan optimization",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_test_execution",
                    parameters={"mode": "fast"},
                    parallel_group="group1"
                )
            ],
            execution_mode=ExecutionMode.PARALLEL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=60.0,
            resource_requirements={}
        )
        
        plan = await workflow_engine._optimize_execution_plan(workflow)
        
        assert isinstance(plan, dict)
        assert "original_steps" in plan
        assert "optimizations" in plan
        assert "parallel_opportunities" in plan
        assert "estimated_savings" in plan
        assert plan["original_steps"] == len(workflow.steps)


class TestEcosystemOptimization:
    """Test system optimization capabilities."""
    
    @pytest.fixture
    async def ecosystem_orchestrator(self):
        """Create ecosystem orchestrator for testing."""
        orchestrator = EcosystemOrchestrator()
        await orchestrator.initialize()
        return orchestrator
    
    @pytest.mark.asyncio
    async def test_optimization_plan_generation(self, ecosystem_orchestrator):
        """Test optimization plan generation for different targets."""
        metrics = SystemPerformanceMetrics(
            timestamp=datetime.now(UTC),
            total_tools_active=20,
            resource_utilization={"cpu": 0.7, "memory": 0.5},
            average_response_time=2.0,
            success_rate=0.85,
            error_rate=0.05,
            throughput=100.0,
            bottlenecks=["cpu_intensive_operations"],
            optimization_opportunities=["cache_optimization"]
        )
        
        # Test performance optimization
        perf_result = await ecosystem_orchestrator.optimize(
            target=OptimizationTarget.PERFORMANCE,
            current_metrics=metrics,
            constraints={"max_resources": {"cpu": 0.8, "memory": 0.7}}
        )
        assert perf_result.is_right()
        perf_plan = perf_result.value
        
        assert perf_plan["optimization_target"] == "performance"
        assert "optimizations" in perf_plan
        assert "estimated_improvement" in perf_plan
        assert perf_plan["estimated_improvement"] > 0.0
        
        # Test efficiency optimization
        eff_result = await ecosystem_orchestrator.optimize(
            target=OptimizationTarget.EFFICIENCY,
            current_metrics=metrics,
            constraints={}
        )
        assert eff_result.is_right()
        eff_plan = eff_result.value
        
        assert eff_plan["optimization_target"] == "efficiency"
        assert len(eff_plan["optimizations"]) > 0
        
        # Test reliability optimization
        rel_result = await ecosystem_orchestrator.optimize(
            target=OptimizationTarget.RELIABILITY,
            current_metrics=metrics,
            constraints={"focus_areas": ["error_handling"]}
        )
        assert rel_result.is_right()
        rel_plan = rel_result.value
        
        assert rel_plan["optimization_target"] == "reliability"
        assert "error_handling_enhancement" in rel_plan["optimizations"]


class TestEcosystemStrategicPlanner:
    """Test strategic automation planning system."""
    
    @pytest.fixture
    def strategic_planner(self):
        """Create strategic planner with tool registry."""
        registry = ToolRegistry()
        
        # Add enterprise and AI tools for testing
        enterprise_tool = ToolDescriptor(
            tool_id="km_enterprise_test",
            tool_name="Enterprise Test Tool",
            category=ToolCategory.ENTERPRISE,
            capabilities={"enterprise_integration"},
            dependencies=[],
            resource_requirements={"cpu": 0.2, "memory": 0.15},
            performance_characteristics={"response_time": 1.0, "reliability": 0.96},
            integration_points=[],
            security_level=SecurityLevel.ENTERPRISE,
            enterprise_ready=True
        )
        registry.register_tool(enterprise_tool)
        
        ai_tool = ToolDescriptor(
            tool_id="km_ai_test",
            tool_name="AI Test Tool",
            category=ToolCategory.INTELLIGENCE,
            capabilities={"ai_processing"},
            dependencies=[],
            resource_requirements={"cpu": 0.4, "memory": 0.3},
            performance_characteristics={"response_time": 2.0, "reliability": 0.90},
            integration_points=[],
            security_level=SecurityLevel.HIGH,
            enterprise_ready=True,
            ai_enhanced=True
        )
        registry.register_tool(ai_tool)
        
        return EcosystemStrategicPlanner(registry)
    
    @pytest.mark.asyncio
    async def test_ecosystem_capabilities_analysis(self, strategic_planner):
        """Test ecosystem capabilities analysis."""
        analysis = await strategic_planner.analyze_ecosystem_capabilities()
        
        assert "total_tools" in analysis
        assert "tools_by_category" in analysis
        assert "enterprise_readiness" in analysis
        assert "ai_enhancement_level" in analysis
        assert "capability_coverage" in analysis
        assert "integration_density" in analysis
        
        assert analysis["total_tools"] == 2
        assert 0.0 <= analysis["enterprise_readiness"] <= 1.0
        assert 0.0 <= analysis["ai_enhancement_level"] <= 1.0
    
    @pytest.mark.asyncio
    async def test_automation_strategy_generation(self, strategic_planner):
        """Test strategic automation plan generation."""
        capabilities = {
            "total_tools": 10,
            "enterprise_readiness": 0.8,
            "ai_enhancement_level": 0.6
        }
        
        objectives = [
            "improve_operational_efficiency",
            "enhance_security_compliance",
            "optimize_resource_utilization"
        ]
        
        strategy = await strategic_planner.generate_automation_strategy(objectives, capabilities)
        
        assert "strategic_objectives" in strategy
        assert "capability_assessment" in strategy
        assert "implementation_roadmap" in strategy
        assert "resource_requirements" in strategy
        assert "expected_benefits" in strategy
        assert "risk_assessment" in strategy
        assert "success_metrics" in strategy
        
        assert strategy["strategic_objectives"] == objectives
        assert "phase_1" in strategy["implementation_roadmap"]
        assert "phase_2" in strategy["implementation_roadmap"]
        assert "phase_3" in strategy["implementation_roadmap"]


class TestEcosystemOrchestrator:
    """Test complete ecosystem orchestrator integration."""
    
    @pytest.fixture
    def ecosystem_orchestrator(self):
        """Create ecosystem orchestrator for testing."""
        return EcosystemOrchestrator()
    
    @pytest.mark.asyncio
    async def test_ecosystem_initialization(self, ecosystem_orchestrator):
        """Test ecosystem initialization."""
        result = await ecosystem_orchestrator.initialize_ecosystem()
        
        # Should succeed with registered tools
        assert result.is_right()
        
        # Verify tool registry is populated
        assert len(ecosystem_orchestrator.tool_registry.tools) > 0
        
        # Verify all components are initialized
        assert ecosystem_orchestrator.workflow_engine is not None
        assert ecosystem_orchestrator.performance_monitor is not None
        assert ecosystem_orchestrator.optimization_engine is not None
        assert ecosystem_orchestrator.strategic_planner is not None
    
    @pytest.mark.asyncio
    async def test_intelligent_workflow_orchestration(self, ecosystem_orchestrator):
        """Test intelligent workflow orchestration."""
        # Initialize ecosystem first
        await ecosystem_orchestrator.initialize_ecosystem()
        
        workflow_spec = {
            "name": "Test Intelligent Workflow",
            "description": "Testing intelligent orchestration",
            "steps": [
                {
                    "tool_id": "km_create_macro",
                    "parameters": {"name": "test_macro", "group": "test_group"},
                    "timeout": 120
                }
            ],
            "execution_mode": "parallel",
            "optimization_target": "performance",
            "expected_duration": 60.0
        }
        
        result = await ecosystem_orchestrator.orchestrate_intelligent_workflow(workflow_spec)
        
        # Should succeed or provide detailed error information
        if result.is_right():
            workflow_result = result.get_right()
            assert "workflow_id" in workflow_result
            assert "execution_mode" in workflow_result
        else:
            error = result.get_left()
            assert isinstance(error, OrchestrationError)
    
    @pytest.mark.asyncio
    async def test_ecosystem_performance_optimization(self, ecosystem_orchestrator):
        """Test ecosystem performance optimization."""
        # Initialize ecosystem first
        await ecosystem_orchestrator.initialize_ecosystem()
        
        result = await ecosystem_orchestrator.optimize_ecosystem_performance(
            OptimizationTarget.PERFORMANCE
        )
        
        assert result.is_right()
        optimization_result = result.get_right()
        
        assert "optimization_target" in optimization_result
        assert "bottlenecks_addressed" in optimization_result
        assert "optimizations_applied" in optimization_result
        assert "performance_improvement" in optimization_result
        assert "current_health_score" in optimization_result
        
        assert optimization_result["optimization_target"] == "performance"
        assert isinstance(optimization_result["bottlenecks_addressed"], list)
        assert isinstance(optimization_result["optimizations_applied"], list)
    
    @pytest.mark.asyncio
    async def test_strategic_automation_planning(self, ecosystem_orchestrator):
        """Test strategic automation planning."""
        # Initialize ecosystem first
        await ecosystem_orchestrator.initialize_ecosystem()
        
        objectives = [
            "optimize_workflow_performance",
            "enhance_enterprise_integration",
            "improve_ai_capabilities"
        ]
        
        result = await ecosystem_orchestrator.generate_strategic_automation_plan(objectives)
        
        assert result.is_right()
        strategic_plan = result.get_right()
        
        assert "strategic_objectives" in strategic_plan
        assert "capability_assessment" in strategic_plan
        assert "implementation_roadmap" in strategic_plan
        assert "resource_requirements" in strategic_plan
        assert "expected_benefits" in strategic_plan
        
        assert strategic_plan["strategic_objectives"] == objectives


class TestUtilityFunctions:
    """Test utility functions for ecosystem orchestration."""
    
    def test_workflow_complexity_calculation(self):
        """Test workflow complexity calculation."""
        simple_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Simple Workflow",
            description="Simple test workflow",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_simple",
                    parameters={}
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=30.0,
            resource_requirements={}
        )
        
        complexity = calculate_workflow_complexity(simple_workflow)
        assert 0.0 <= complexity <= 10.0
        assert complexity < 1.0  # Simple workflow should have low complexity
        
        complex_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Complex Workflow",
            description="Complex test workflow",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_complex1",
                    parameters={},
                    parallel_group="group1"
                ),
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_complex2",
                    parameters={},
                    parallel_group="group2",
                    preconditions=["condition1"],
                    postconditions=["result1"]
                ),
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_complex3",
                    parameters={},
                    preconditions=["result1"],
                    postconditions=["final_result"]
                )
            ],
            execution_mode=ExecutionMode.ADAPTIVE,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=180.0,
            resource_requirements={"cpu": 0.8, "memory": 0.6, "network": 0.4}
        )
        
        complex_complexity = calculate_workflow_complexity(complex_workflow)
        assert complex_complexity > complexity  # Complex workflow should have higher complexity
    
    def test_workflow_duration_estimation(self):
        """Test workflow duration estimation."""
        registry = ToolRegistry()
        
        # Register test tool with performance characteristics
        test_tool = ToolDescriptor(
            tool_id="km_duration_test",
            tool_name="Duration Test Tool",
            category=ToolCategory.FOUNDATION,
            capabilities={"test_duration"},
            dependencies=[],
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            performance_characteristics={"response_time": 2.0, "reliability": 0.95},
            integration_points=[],
            security_level=SecurityLevel.STANDARD
        )
        registry.register_tool(test_tool)
        
        workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Duration Test Workflow",
            description="Test duration estimation",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_duration_test",
                    parameters={},
                    timeout=300
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.PERFORMANCE,
            expected_duration=60.0,
            resource_requirements={}
        )
        
        estimated_duration = estimate_workflow_duration(workflow, registry)
        assert estimated_duration > 0.0
        assert estimated_duration >= 2.0  # Should be at least the tool's response time
    
    def test_workflow_security_validation(self):
        """Test workflow security validation."""
        registry = ToolRegistry()
        
        # Register tools with different security levels
        standard_tool = ToolDescriptor(
            tool_id="km_standard_security",
            tool_name="Standard Security Tool",
            category=ToolCategory.FOUNDATION,
            capabilities={"standard_operations"},
            dependencies=[],
            resource_requirements={"cpu": 0.1, "memory": 0.05},
            performance_characteristics={"response_time": 1.0, "reliability": 0.95},
            integration_points=[],
            security_level=SecurityLevel.STANDARD
        )
        registry.register_tool(standard_tool)
        
        enterprise_tool = ToolDescriptor(
            tool_id="km_enterprise_security",
            tool_name="Enterprise Security Tool",
            category=ToolCategory.ENTERPRISE,
            capabilities={"enterprise_operations"},
            dependencies=[],
            resource_requirements={"cpu": 0.2, "memory": 0.1},
            performance_characteristics={"response_time": 1.5, "reliability": 0.97},
            integration_points=[],
            security_level=SecurityLevel.ENTERPRISE
        )
        registry.register_tool(enterprise_tool)
        
        # Valid workflow with consistent security levels
        valid_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Valid Security Workflow",
            description="Test security validation",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_standard_security",
                    parameters={"operation": "safe_operation"}
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.RELIABILITY,
            expected_duration=30.0,
            resource_requirements={}
        )
        
        result = validate_workflow_security(valid_workflow, registry)
        assert result.is_right()
        
        # Invalid workflow with security escalation
        invalid_workflow = EcosystemWorkflow(
            workflow_id=create_workflow_id(),
            name="Invalid Security Workflow",
            description="Test security validation",
            steps=[
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_standard_security",
                    parameters={"operation": "safe_operation"}
                ),
                WorkflowStep(
                    step_id=create_step_id(),
                    tool_id="km_enterprise_security",
                    parameters={"operation": "enterprise_operation"}
                )
            ],
            execution_mode=ExecutionMode.SEQUENTIAL,
            optimization_target=OptimizationTarget.RELIABILITY,
            expected_duration=60.0,
            resource_requirements={}
        )
        
        result = validate_workflow_security(invalid_workflow, registry)
        assert result.is_left()
        assert isinstance(result.get_left(), OrchestrationError)


class TestOrchestrationErrors:
    """Test orchestration error handling and validation."""
    
    def test_orchestration_error_creation(self):
        """Test orchestration error creation and messaging."""
        # Workflow execution failure
        workflow_error = OrchestrationError.workflow_execution_failed("Test failure reason")
        assert "workflow_execution" in str(workflow_error)
        assert "Test failure reason" in str(workflow_error)
        
        # Tool not found error
        tool_error = OrchestrationError.tool_not_found("km_missing_tool")
        assert "tool_id" in str(tool_error)
        assert "km_missing_tool" in str(tool_error)
        
        # Security escalation error
        security_error = OrchestrationError.security_escalation_detected()
        assert "security_escalation" in str(security_error)
        
        # Optimization failure error
        optimization_error = OrchestrationError.optimization_failed("Optimization test failure")
        assert "optimization" in str(optimization_error)
        assert "Optimization test failure" in str(optimization_error)


@pytest.mark.asyncio
async def test_complete_orchestration_integration():
    """Integration test for complete orchestration workflow."""
    # Create and initialize orchestrator
    orchestrator = EcosystemOrchestrator()
    init_result = await orchestrator.initialize_ecosystem()
    assert init_result.is_right()
    
    # Test workflow specification
    workflow_spec = {
        "name": "Integration Test Workflow",
        "description": "Complete integration test",
        "steps": [
            {
                "tool_id": "km_create_macro",
                "parameters": {"name": "integration_test", "group": "test"},
                "timeout": 120
            }
        ],
        "execution_mode": "sequential",
        "optimization_target": "reliability",
        "expected_duration": 90.0
    }
    
    # Test orchestration
    orchestration_result = await orchestrator.orchestrate_intelligent_workflow(workflow_spec)
    
    # Test optimization
    optimization_result = await orchestrator.optimize_ecosystem_performance(OptimizationTarget.EFFICIENCY)
    assert optimization_result.is_right()
    
    # Test strategic planning
    strategic_result = await orchestrator.generate_strategic_automation_plan([
        "test_objective_1", "test_objective_2"
    ])
    assert strategic_result.is_right()
    
    # Verify all components work together
    final_metrics = await orchestrator.performance_monitor.get_current_metrics()
    assert isinstance(final_metrics, SystemPerformanceMetrics)
    assert final_metrics.get_health_score() >= 0.0


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])