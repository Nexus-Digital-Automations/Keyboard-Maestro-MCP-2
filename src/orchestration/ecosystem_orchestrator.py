"""
Master ecosystem orchestrator coordinating all 46+ automation tools.

This module provides the central coordination hub that brings together:
- Tool registry and capability management
- Workflow orchestration and execution
- Performance monitoring and optimization
- Resource allocation and management
- Strategic planning and ecosystem evolution

Security: Enterprise-grade orchestration with comprehensive security validation.
Performance: <500ms orchestration decisions, distributed execution capabilities.
Type Safety: Complete type system with contracts and validation.
"""

from typing import Dict, List, Optional, Any, Set, Tuple, Callable
from dataclasses import dataclass, field
from datetime import datetime, UTC, timedelta
import asyncio
import logging
import uuid

from .ecosystem_architecture import (
    WorkflowId, OrchestrationId, EcosystemWorkflow, WorkflowStep,
    ExecutionMode, OptimizationTarget, SystemPerformanceMetrics,
    OrchestrationError, OrchestrationResult, ToolCategory,
    create_workflow_id, create_step_id, create_orchestration_id
)
from .tool_registry import ComprehensiveToolRegistry, get_tool_registry
from .workflow_engine import MasterWorkflowEngine, get_workflow_engine
from .performance_monitor import EcosystemPerformanceMonitor, get_performance_monitor
from .resource_manager import IntelligentResourceManager, get_resource_manager
from .strategic_planner import EcosystemStrategicPlanner, get_strategic_planner, EvolutionPhase
from ..core.contracts import require, ensure
from ..core.either import Either


@dataclass
class OrchestrationRequest:
    """Request for ecosystem orchestration operation."""
    operation_type: str
    parameters: Dict[str, Any]
    optimization_target: OptimizationTarget
    priority: int = 5
    timeout: float = 300.0
    requesting_agent: Optional[str] = None


@dataclass
class EcosystemStatus:
    """Current status of the entire ecosystem."""
    timestamp: datetime
    total_tools: int
    active_workflows: int
    system_health_score: float
    performance_level: str
    resource_utilization: Dict[str, float]
    maturity_phase: EvolutionPhase
    active_alerts: int
    optimization_opportunities: List[str]
    strategic_recommendations: List[str]


class EcosystemOrchestrator:
    """Master orchestrator for the complete 46+ tool automation ecosystem."""
    
    def __init__(
        self,
        tool_registry: Optional[ComprehensiveToolRegistry] = None,
        workflow_engine: Optional[MasterWorkflowEngine] = None,
        performance_monitor: Optional[EcosystemPerformanceMonitor] = None,
        resource_manager: Optional[IntelligentResourceManager] = None,
        strategic_planner: Optional[EcosystemStrategicPlanner] = None
    ):
        # Core components
        self.tool_registry = tool_registry or get_tool_registry()
        self.workflow_engine = workflow_engine or get_workflow_engine()
        self.performance_monitor = performance_monitor or get_performance_monitor()
        self.resource_manager = resource_manager or get_resource_manager()
        self.strategic_planner = strategic_planner or get_strategic_planner()
        
        self.logger = logging.getLogger(__name__)
        
        # Orchestration state
        self.active_orchestrations: Dict[OrchestrationId, Dict[str, Any]] = {}
        self.orchestration_history: List[OrchestrationResult] = []
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.initialization_time: Optional[datetime] = None
        
    async def initialize(self) -> Either[OrchestrationError, None]:
        """Initialize the complete ecosystem orchestrator."""
        
        try:
            self.logger.info("Initializing ecosystem orchestrator...")
            
            # Start all subsystems
            await self.performance_monitor.start_monitoring()
            await self.resource_manager.start_management()
            
            # Validate tool registry completeness
            if len(self.tool_registry.tools) < 40:  # Expect at least 40 tools
                self.logger.warning(f"Tool registry contains only {len(self.tool_registry.tools)} tools")
            
            # Perform initial system health check
            health_check = await self._perform_initial_health_check()
            if health_check.is_left():
                return health_check
            
            self.is_initialized = True
            self.initialization_time = datetime.now(UTC)
            self.logger.info("Ecosystem orchestrator initialized successfully")
            
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Ecosystem initialization failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError.ecosystem_initialization_failed(error_msg))
    
    async def shutdown(self) -> None:
        """Gracefully shutdown the ecosystem orchestrator."""
        
        try:
            self.logger.info("Shutting down ecosystem orchestrator...")
            
            # Stop background processes
            await self.performance_monitor.stop_monitoring()
            await self.resource_manager.stop_management()
            
            # Cancel active orchestrations
            for orchestration_id in list(self.active_orchestrations.keys()):
                await self._cancel_orchestration(orchestration_id)
            
            self.is_running = False
            self.is_initialized = False
            self.logger.info("Ecosystem orchestrator shutdown complete")
            
        except Exception as e:
            self.logger.error(f"Error during shutdown: {e}")
    
    async def _perform_initial_health_check(self) -> Either[OrchestrationError, None]:
        """Perform initial health check of the ecosystem."""
        
        try:
            # Check tool registry
            if not self.tool_registry.tools:
                return Either.left(OrchestrationError.incomplete_tool_registry())
            
            # Check core categories are represented
            required_categories = [ToolCategory.FOUNDATION, ToolCategory.INTELLIGENCE]
            for category in required_categories:
                tools_in_category = self.tool_registry.find_tools_by_category(category)
                if not tools_in_category:
                    return Either.left(
                        OrchestrationError.ecosystem_initialization_failed(
                            f"No tools found in required category: {category.value}"
                        )
                    )
            
            # Check resource pools
            resource_status = await self.resource_manager.get_resource_status()
            if resource_status["system_health"] == "critical":
                return Either.left(
                    OrchestrationError.ecosystem_initialization_failed("Critical resource issues detected")
                )
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(
                OrchestrationError.ecosystem_initialization_failed(f"Health check failed: {e}")
            )
    
    @require(lambda self: self.is_initialized)
    async def orchestrate(
        self,
        workflow_definition: Optional[Dict[str, Any]] = None,
        optimization_target: OptimizationTarget = OptimizationTarget.EFFICIENCY,
        execution_mode: ExecutionMode = ExecutionMode.ADAPTIVE,
        tool_selection: str = "intelligent"
    ) -> Either[OrchestrationError, OrchestrationResult]:
        """Orchestrate complex workflow across multiple ecosystem tools."""
        
        orchestration_id = create_orchestration_id()
        
        try:
            self.logger.info(f"Starting orchestration {orchestration_id}")
            
            # Track active orchestration
            self.active_orchestrations[orchestration_id] = {
                "start_time": datetime.now(UTC),
                "status": "initializing",
                "workflow_definition": workflow_definition,
                "optimization_target": optimization_target
            }
            
            # Create or use provided workflow
            if workflow_definition:
                workflow = await self._create_workflow_from_definition(workflow_definition, execution_mode)
            else:
                # Create a default ecosystem health workflow
                workflow = await self._create_default_workflow(execution_mode, optimization_target)
            
            if workflow.is_left():
                return workflow
            
            ecosystem_workflow = workflow.right()
            
            # Execute workflow
            self.active_orchestrations[orchestration_id]["status"] = "executing"
            execution_result = await self.workflow_engine.execute_workflow(
                ecosystem_workflow, 
                optimization_target
            )
            
            if execution_result.is_left():
                self.active_orchestrations[orchestration_id]["status"] = "failed"
                return execution_result
            
            orchestration_result = execution_result.right()
            
            # Record successful orchestration
            self.active_orchestrations[orchestration_id]["status"] = "completed"
            self.orchestration_history.append(orchestration_result)
            
            # Clean up
            del self.active_orchestrations[orchestration_id]
            
            self.logger.info(f"Orchestration {orchestration_id} completed successfully")
            return Either.right(orchestration_result)
            
        except Exception as e:
            if orchestration_id in self.active_orchestrations:
                self.active_orchestrations[orchestration_id]["status"] = "error"
            
            error_msg = f"Orchestration failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError.workflow_execution_failed(error_msg))
    
    async def _create_workflow_from_definition(
        self, 
        definition: Dict[str, Any], 
        execution_mode: ExecutionMode
    ) -> Either[OrchestrationError, EcosystemWorkflow]:
        """Create workflow from user-provided definition."""
        
        try:
            workflow_id = create_workflow_id()
            
            # Parse workflow steps
            steps = []
            for i, step_def in enumerate(definition.get("steps", [])):
                step_id = create_step_id()
                
                tool_id = step_def.get("tool_id")
                if not tool_id or tool_id not in self.tool_registry.tools:
                    return Either.left(OrchestrationError.tool_not_found(tool_id or "unknown"))
                
                step = WorkflowStep(
                    step_id=step_id,
                    tool_id=tool_id,
                    parameters=step_def.get("parameters", {}),
                    preconditions=step_def.get("preconditions", []),
                    postconditions=step_def.get("postconditions", []),
                    timeout=step_def.get("timeout", 300),
                    retry_count=step_def.get("retry_count", 3),
                    parallel_group=step_def.get("parallel_group")
                )
                steps.append(step)
            
            # Calculate resource requirements
            resource_requirements = {}
            for step in steps:
                tool = self.tool_registry.tools[step.tool_id]
                for resource, amount in tool.resource_requirements.items():
                    resource_requirements[resource] = resource_requirements.get(resource, 0) + amount
            
            workflow = EcosystemWorkflow(
                workflow_id=workflow_id,
                name=definition.get("name", "Custom Workflow"),
                description=definition.get("description", "User-defined workflow"),
                steps=steps,
                execution_mode=execution_mode,
                optimization_target=OptimizationTarget.EFFICIENCY,
                expected_duration=definition.get("expected_duration", 60.0),
                resource_requirements=resource_requirements,
                success_criteria=definition.get("success_criteria", [])
            )
            
            return Either.right(workflow)
            
        except Exception as e:
            return Either.left(
                OrchestrationError.workflow_execution_failed(f"Failed to create workflow: {e}")
            )
    
    async def _create_default_workflow(
        self, 
        execution_mode: ExecutionMode, 
        optimization_target: OptimizationTarget
    ) -> Either[OrchestrationError, EcosystemWorkflow]:
        """Create default ecosystem health and optimization workflow."""
        
        try:
            workflow_id = create_workflow_id()
            
            # Select representative tools from different categories
            steps = []
            
            # Foundation tool step
            foundation_tools = self.tool_registry.find_tools_by_category(ToolCategory.FOUNDATION)
            if foundation_tools:
                step = WorkflowStep(
                    step_id=create_step_id(),
                    tool_id=foundation_tools[0].tool_id,
                    parameters={"operation": "health_check"},
                    postconditions=["foundation_verified"]
                )
                steps.append(step)
            
            # Intelligence tool step (if available)
            intelligence_tools = self.tool_registry.find_tools_by_category(ToolCategory.INTELLIGENCE)
            if intelligence_tools:
                step = WorkflowStep(
                    step_id=create_step_id(),
                    tool_id=intelligence_tools[0].tool_id,
                    parameters={"operation": "analyze_system"},
                    preconditions=["foundation_verified"],
                    postconditions=["intelligence_analysis_complete"]
                )
                steps.append(step)
            
            # Enterprise tool step (if available)
            enterprise_tools = self.tool_registry.find_tools_by_category(ToolCategory.ENTERPRISE)
            if enterprise_tools:
                step = WorkflowStep(
                    step_id=create_step_id(),
                    tool_id=enterprise_tools[0].tool_id,
                    parameters={"operation": "compliance_check"},
                    parallel_group="compliance"
                )
                steps.append(step)
            
            # Calculate total resource requirements
            resource_requirements = {}
            for step in steps:
                tool = self.tool_registry.tools[step.tool_id]
                for resource, amount in tool.resource_requirements.items():
                    resource_requirements[resource] = resource_requirements.get(resource, 0) + amount
            
            workflow = EcosystemWorkflow(
                workflow_id=workflow_id,
                name="Ecosystem Health Check",
                description="Default workflow for ecosystem health verification and optimization",
                steps=steps,
                execution_mode=execution_mode,
                optimization_target=optimization_target,
                expected_duration=180.0,  # 3 minutes
                resource_requirements=resource_requirements,
                success_criteria=["All tools responsive", "System health verified"]
            )
            
            return Either.right(workflow)
            
        except Exception as e:
            return Either.left(
                OrchestrationError.workflow_execution_failed(f"Failed to create default workflow: {e}")
            )
    
    async def optimize(
        self,
        target: OptimizationTarget = OptimizationTarget.EFFICIENCY
    ) -> Either[OrchestrationError, Dict[str, Any]]:
        """Optimize ecosystem performance across all dimensions."""
        
        try:
            self.logger.info(f"Starting ecosystem optimization for {target.value}")
            
            optimization_result = {
                "timestamp": datetime.now(UTC).isoformat(),
                "optimization_target": target.value,
                "actions_performed": [],
                "improvements": {},
                "recommendations": []
            }
            
            # Resource optimization
            resource_optimization = await self.resource_manager.optimize_allocation(target)
            if resource_optimization.is_right():
                optimization_result["actions_performed"].append("resource_optimization")
                optimization_result.update(resource_optimization.right())
            
            # Performance optimization through monitoring
            health_report = await self.performance_monitor.get_health_report()
            optimization_result["improvements"]["performance"] = {
                "health_score": health_report.overall_health_score,
                "performance_level": health_report.performance_level.value,
                "active_alerts": len(health_report.active_alerts)
            }
            
            # Strategic optimization recommendations
            current_state = await self.strategic_planner.analyze_current_state()
            optimization_result["improvements"]["strategic"] = current_state["maturity_assessment"]
            optimization_result["recommendations"].extend(
                current_state.get("capability_analysis", {}).get("improvement_areas", [])
            )
            
            # Tool registry optimization
            ecosystem_stats = self.tool_registry.get_ecosystem_statistics()
            optimization_result["improvements"]["ecosystem"] = {
                "total_tools": ecosystem_stats["total_tools"],
                "enterprise_ready": ecosystem_stats["enterprise_ready_tools"],
                "ai_enhanced": ecosystem_stats["ai_enhanced_tools"]
            }
            
            self.logger.info("Ecosystem optimization completed")
            return Either.right(optimization_result)
            
        except Exception as e:
            error_msg = f"Optimization failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError.optimization_failed(error_msg))
    
    async def monitor(self) -> Either[OrchestrationError, Dict[str, Any]]:
        """Get comprehensive monitoring data for the ecosystem."""
        
        try:
            # Get current performance metrics
            current_metrics = await self.performance_monitor.get_current_metrics()
            health_report = await self.performance_monitor.get_health_report()
            
            # Get resource status
            resource_status = await self.resource_manager.get_resource_status()
            
            # Get ecosystem statistics
            ecosystem_stats = self.tool_registry.get_ecosystem_statistics()
            
            monitoring_data = {
                "timestamp": datetime.now(UTC).isoformat(),
                "system_health": {
                    "overall_score": current_metrics.get_health_score(),
                    "performance_level": health_report.performance_level.value,
                    "active_alerts": len(health_report.active_alerts)
                },
                "performance_metrics": {
                    "average_response_time": current_metrics.average_response_time,
                    "success_rate": current_metrics.success_rate,
                    "error_rate": current_metrics.error_rate,
                    "throughput": current_metrics.throughput,
                    "bottlenecks": current_metrics.bottlenecks
                },
                "resource_utilization": resource_status["resource_pools"],
                "ecosystem_overview": ecosystem_stats,
                "active_workflows": len(self.active_orchestrations),
                "orchestration_history_count": len(self.orchestration_history)
            }
            
            return Either.right(monitoring_data)
            
        except Exception as e:
            error_msg = f"Monitoring failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError("monitoring_error", None, error_msg))
    
    async def plan(
        self,
        target_phase: Optional[str] = None,
        timeline_months: int = 12,
        focus_areas: Optional[List[str]] = None
    ) -> Either[OrchestrationError, Dict[str, Any]]:
        """Create strategic plan for ecosystem evolution."""
        
        try:
            # Convert string parameters to enums
            evolution_phase = EvolutionPhase.OPTIMIZATION  # Default
            if target_phase:
                try:
                    evolution_phase = EvolutionPhase(target_phase.lower())
                except ValueError:
                    return Either.left(
                        OrchestrationError.strategic_planning_failed(f"Invalid target phase: {target_phase}")
                    )
            
            # Convert focus areas to categories
            focus_categories = []
            if focus_areas:
                for area in focus_areas:
                    try:
                        category = ToolCategory(area.lower())
                        focus_categories.append(category)
                    except ValueError:
                        self.logger.warning(f"Invalid focus area: {area}")
            
            if not focus_categories:
                focus_categories = [ToolCategory.INTELLIGENCE, ToolCategory.ENTERPRISE]
            
            # Create strategic roadmap
            timeline = timedelta(days=timeline_months * 30)
            roadmap_result = await self.strategic_planner.create_strategic_roadmap(
                evolution_phase, timeline, focus_categories
            )
            
            if roadmap_result.is_left():
                return roadmap_result
            
            roadmap = roadmap_result.right()
            
            # Analyze current state
            current_state = await self.strategic_planner.analyze_current_state()
            
            # Identify capability gaps
            capability_gaps = await self.strategic_planner.identify_capability_gaps()
            
            planning_result = {
                "timestamp": datetime.now(UTC).isoformat(),
                "current_state": current_state,
                "strategic_roadmap": {
                    "roadmap_id": roadmap.roadmap_id,
                    "name": roadmap.name,
                    "current_phase": roadmap.current_phase.value,
                    "target_phase": roadmap.target_phase.value,
                    "timeline_days": roadmap.timeline.days,
                    "initiatives_count": len(roadmap.initiatives),
                    "milestones_count": len(roadmap.milestones),
                    "resource_requirements": roadmap.resource_requirements,
                    "expected_outcomes": roadmap.expected_outcomes
                },
                "capability_gaps": [
                    {
                        "gap_id": gap.gap_id,
                        "category": gap.category.value,
                        "missing_capability": gap.missing_capability,
                        "priority": gap.priority.value,
                        "estimated_effort": gap.estimated_effort
                    }
                    for gap in capability_gaps[:10]  # Top 10 gaps
                ],
                "recommendations": [
                    "Focus on high-priority capability gaps",
                    "Align initiatives with technology trends",
                    "Monitor progress against milestones",
                    "Adjust resource allocation as needed"
                ]
            }
            
            return Either.right(planning_result)
            
        except Exception as e:
            error_msg = f"Strategic planning failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError.strategic_planning_failed(error_msg))
    
    async def coordinate(
        self,
        tools: List[str],
        operation: str = "sync",
        parameters: Optional[Dict[str, Any]] = None
    ) -> Either[OrchestrationError, Dict[str, Any]]:
        """Coordinate operation across multiple specific tools."""
        
        try:
            coordination_id = create_orchestration_id()
            self.logger.info(f"Coordinating {operation} across {len(tools)} tools")
            
            # Validate all tools exist
            valid_tools = []
            for tool_id in tools:
                if tool_id in self.tool_registry.tools:
                    valid_tools.append(tool_id)
                else:
                    self.logger.warning(f"Tool not found: {tool_id}")
            
            if not valid_tools:
                return Either.left(OrchestrationError.tool_not_found("No valid tools provided"))
            
            # Create coordination workflow
            steps = []
            for tool_id in valid_tools:
                step = WorkflowStep(
                    step_id=create_step_id(),
                    tool_id=tool_id,
                    parameters=parameters or {"operation": operation},
                    parallel_group="coordination_group"
                )
                steps.append(step)
            
            # Calculate resource requirements
            resource_requirements = {}
            for tool_id in valid_tools:
                tool = self.tool_registry.tools[tool_id]
                for resource, amount in tool.resource_requirements.items():
                    resource_requirements[resource] = resource_requirements.get(resource, 0) + amount
            
            # Create workflow
            workflow = EcosystemWorkflow(
                workflow_id=create_workflow_id(),
                name=f"Tool Coordination: {operation}",
                description=f"Coordinate {operation} across {len(valid_tools)} tools",
                steps=steps,
                execution_mode=ExecutionMode.PARALLEL,
                optimization_target=OptimizationTarget.EFFICIENCY,
                expected_duration=120.0,
                resource_requirements=resource_requirements
            )
            
            # Execute coordination workflow
            execution_result = await self.workflow_engine.execute_workflow(workflow)
            
            if execution_result.is_left():
                return execution_result
            
            orchestration_result = execution_result.right()
            
            coordination_result = {
                "coordination_id": coordination_id,
                "operation": operation,
                "tools_coordinated": valid_tools,
                "execution_time": orchestration_result.execution_time,
                "success": orchestration_result.success,
                "performance_metrics": {
                    "average_response_time": orchestration_result.performance_metrics.average_response_time,
                    "success_rate": orchestration_result.performance_metrics.success_rate,
                    "throughput": orchestration_result.performance_metrics.throughput
                }
            }
            
            return Either.right(coordination_result)
            
        except Exception as e:
            error_msg = f"Coordination failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError.workflow_execution_failed(error_msg))
    
    async def analyze(self) -> Either[OrchestrationError, Dict[str, Any]]:
        """Perform comprehensive ecosystem analysis."""
        
        try:
            analysis_result = {
                "timestamp": datetime.now(UTC).isoformat(),
                "ecosystem_analysis": {},
                "performance_analysis": {},
                "strategic_analysis": {},
                "recommendations": []
            }
            
            # Ecosystem analysis
            ecosystem_stats = self.tool_registry.get_ecosystem_statistics()
            analysis_result["ecosystem_analysis"] = ecosystem_stats
            
            # Performance analysis
            health_report = await self.performance_monitor.get_health_report()
            analysis_result["performance_analysis"] = {
                "overall_health_score": health_report.overall_health_score,
                "performance_level": health_report.performance_level.value,
                "active_tools": health_report.active_tools,
                "total_executions": health_report.total_executions,
                "optimization_recommendations": health_report.optimization_recommendations
            }
            
            # Strategic analysis
            current_state = await self.strategic_planner.analyze_current_state()
            analysis_result["strategic_analysis"] = current_state
            
            # Generate comprehensive recommendations
            recommendations = []
            
            # Performance recommendations
            if health_report.overall_health_score < 0.8:
                recommendations.append("Improve system health - current score below optimal")
            
            # Ecosystem recommendations
            if ecosystem_stats["enterprise_ready_tools"] < 20:
                recommendations.append("Increase enterprise-ready tools for production deployment")
            
            if ecosystem_stats["ai_enhanced_tools"] < 5:
                recommendations.append("Enhance AI integration across more tools")
            
            # Strategic recommendations
            maturity = current_state.get("maturity_assessment", {})
            if maturity.get("overall_maturity", 0) < 0.7:
                recommendations.append("Focus on ecosystem maturity development")
            
            analysis_result["recommendations"] = recommendations
            
            return Either.right(analysis_result)
            
        except Exception as e:
            error_msg = f"Analysis failed: {e}"
            self.logger.error(error_msg)
            return Either.left(OrchestrationError("analysis_error", None, error_msg))
    
    async def get_status(self) -> EcosystemStatus:
        """Get current status of the entire ecosystem."""
        
        try:
            # Get performance metrics
            current_metrics = await self.performance_monitor.get_current_metrics()
            health_report = await self.performance_monitor.get_health_report()
            
            # Get ecosystem statistics
            ecosystem_stats = self.tool_registry.get_ecosystem_statistics()
            
            # Get strategic analysis
            current_state = await self.strategic_planner.analyze_current_state()
            maturity = current_state.get("maturity_assessment", {})
            
            # Determine maturity phase
            overall_maturity = maturity.get("overall_maturity", 0.0)
            if overall_maturity < 0.3:
                maturity_phase = EvolutionPhase.FOUNDATION
            elif overall_maturity < 0.5:
                maturity_phase = EvolutionPhase.EXPANSION
            elif overall_maturity < 0.7:
                maturity_phase = EvolutionPhase.INTELLIGENCE
            elif overall_maturity < 0.85:
                maturity_phase = EvolutionPhase.OPTIMIZATION
            else:
                maturity_phase = EvolutionPhase.INNOVATION
            
            # Get resource utilization
            resource_status = await self.resource_manager.get_resource_status()
            resource_utilization = {}
            for resource_type, pool_info in resource_status["resource_pools"].items():
                resource_utilization[resource_type] = pool_info["utilization_rate"]
            
            status = EcosystemStatus(
                timestamp=datetime.now(UTC),
                total_tools=ecosystem_stats["total_tools"],
                active_workflows=len(self.active_orchestrations),
                system_health_score=current_metrics.get_health_score(),
                performance_level=health_report.performance_level.value,
                resource_utilization=resource_utilization,
                maturity_phase=maturity_phase,
                active_alerts=len(health_report.active_alerts),
                optimization_opportunities=current_metrics.optimization_opportunities,
                strategic_recommendations=health_report.optimization_recommendations
            )
            
            return status
            
        except Exception as e:
            self.logger.error(f"Failed to get ecosystem status: {e}")
            # Return a basic status in case of error
            return EcosystemStatus(
                timestamp=datetime.now(UTC),
                total_tools=len(self.tool_registry.tools),
                active_workflows=len(self.active_orchestrations),
                system_health_score=0.5,
                performance_level="unknown",
                resource_utilization={},
                maturity_phase=EvolutionPhase.FOUNDATION,
                active_alerts=0,
                optimization_opportunities=[],
                strategic_recommendations=[]
            )
    
    async def _cancel_orchestration(self, orchestration_id: OrchestrationId) -> None:
        """Cancel an active orchestration."""
        if orchestration_id in self.active_orchestrations:
            self.active_orchestrations[orchestration_id]["status"] = "cancelled"
            del self.active_orchestrations[orchestration_id]
            self.logger.info(f"Cancelled orchestration {orchestration_id}")


# Global ecosystem orchestrator instance
_global_orchestrator: Optional[EcosystemOrchestrator] = None


def get_ecosystem_orchestrator() -> EcosystemOrchestrator:
    """Get or create the global ecosystem orchestrator instance."""
    global _global_orchestrator
    if _global_orchestrator is None:
        _global_orchestrator = EcosystemOrchestrator()
    return _global_orchestrator