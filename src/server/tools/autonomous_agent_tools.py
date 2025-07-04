"""
FastMCP tools for autonomous agent creation and management.

This module provides comprehensive tools for creating and managing self-managing
automation agents with learning capabilities, goal-driven behavior, and
intelligent resource optimization.

Security: All agent operations include comprehensive safety validation and human oversight options
Performance: <5s agent initialization, <10s decision-making, <60s autonomous cycles
Enterprise: Full audit integration, multi-agent coordination, and scalable architecture
"""

import logging
from typing import Dict, List, Optional, Any, Union
from datetime import datetime, timedelta, UTC
import asyncio
from dataclasses import asdict

from fastmcp import Context
from fastmcp.exceptions import ToolError as InternalError, ValidationError as InvalidParams

from ...core.autonomous_systems import (
    AgentId, GoalId, AgentType, AutonomyLevel, AgentStatus,
    AgentGoal, GoalPriority, AgentConfiguration, ActionType,
    AutonomousAgentError, create_goal_id, get_default_config,
    ConfidenceScore, RiskScore, PerformanceMetric
)
from ...core.either import Either
from ...agents.agent_manager import AgentManager, AutonomousAgent
# from ...core.ai_integration import AIProcessor  # Not implemented yet
from ...ai.model_manager import AIModelManager as ModelManager
from ...audit.audit_system_manager import AuditSystemManager, AuditEventType


# Singleton agent manager instance
_agent_manager: Optional[AgentManager] = None
# _ai_processor: Optional[AIProcessor] = None  # Not implemented yet
_model_manager: Optional[ModelManager] = None
_audit_manager: Optional[AuditSystemManager] = None


def get_agent_manager() -> AgentManager:
    """Get or create singleton agent manager instance."""
    global _agent_manager
    if _agent_manager is None:
        _agent_manager = AgentManager()
    return _agent_manager


# def get_ai_processor() -> Optional[AIProcessor]:
#     """Get AI processor if available."""
#     global _ai_processor
#     if _ai_processor is None:
#         try:
#             _ai_processor = AIProcessor()
#         except Exception:
#             # AI processor not available
#             pass
#     return _ai_processor


def get_model_manager() -> Optional[ModelManager]:
    """Get model manager if available."""
    global _model_manager
    if _model_manager is None:
        try:
            _model_manager = ModelManager()
        except Exception:
            # Model manager not available
            pass
    return _model_manager


def get_audit_manager() -> Optional[AuditSystemManager]:
    """Get audit manager if available."""
    global _audit_manager
    if _audit_manager is None:
        try:
            _audit_manager = AuditSystemManager()
        except Exception:
            # Audit manager not available
            pass
    return _audit_manager


async def _log_agent_operation(
    operation: str,
    agent_id: Optional[str] = None,
    details: Optional[Dict[str, Any]] = None,
    success: bool = True
) -> None:
    """Log agent operation to audit system."""
    audit_manager = get_audit_manager()
    if not audit_manager:
        return
    
    event_data = {
        "operation": operation,
        "success": success,
        "timestamp": datetime.now(UTC).isoformat()
    }
    
    if agent_id:
        event_data["agent_id"] = agent_id
    
    if details:
        event_data.update(details)
    
    await audit_manager.log_event(
        event_type=AuditEventType.DATA_ACCESS if operation == "monitor" else AuditEventType.CONFIGURATION_CHANGE,
        event_data=event_data,
        user_id="autonomous_agent_system",
        severity="INFO" if success else "WARNING"
    )


async def km_autonomous_agent(
    operation: str,
    agent_type: str = "general",
    agent_config: Optional[Dict[str, Any]] = None,
    goals: Optional[List[Dict[str, Any]]] = None,
    learning_mode: bool = True,
    autonomy_level: str = "supervised",
    resource_limits: Optional[Dict[str, float]] = None,
    safety_constraints: Optional[Dict[str, Any]] = None,
    communication_enabled: bool = True,
    monitoring_interval: int = 60,
    optimization_frequency: str = "hourly",
    human_approval_required: bool = False,
    timeout: int = 300,
    ctx: Optional[Context] = None
) -> Dict[str, Any]:
    """
    Create and manage autonomous agents with learning capabilities.
    
    Args:
        operation: Operation to perform (create|start|stop|configure|monitor|optimize)
        agent_type: Type of agent (general|optimizer|monitor|learner|coordinator)
        agent_config: Custom agent configuration parameters
        goals: List of goals for the agent to pursue
        learning_mode: Enable learning and adaptation
        autonomy_level: Level of autonomy (manual|supervised|autonomous|full)
        resource_limits: Resource usage limits (CPU, memory, etc.)
        safety_constraints: Safety rules and constraints
        communication_enabled: Enable inter-agent communication
        monitoring_interval: Self-monitoring interval in seconds
        optimization_frequency: Optimization frequency (never|hourly|daily|weekly|adaptive)
        human_approval_required: Require human approval for actions
        timeout: Operation timeout in seconds
        ctx: FastMCP context for progress reporting
        
    Returns:
        Dict containing operation results and agent status
    """
    try:
        if ctx:
            await ctx.report_progress(0, 100, f"Starting autonomous agent operation: {operation}")
        
        # Validate operation
        valid_operations = ["create", "start", "stop", "configure", "monitor", "optimize", "add_goal", "status", "list"]
        if operation not in valid_operations:
            raise InvalidParams(f"Invalid operation. Must be one of: {', '.join(valid_operations)}")
        
        manager = get_agent_manager()
        
        # Handle list operation
        if operation == "list":
            agents = manager.list_agents()
            
            if ctx:
                await ctx.report_progress(100, 100, f"Listed {len(agents)} agents")
            
            await _log_agent_operation("list", details={"agent_count": len(agents)})
            
            return {
                "success": True,
                "operation": "list",
                "agents": agents,
                "total_count": len(agents)
            }
        
        # Handle create operation
        if operation == "create":
            # Validate agent type
            try:
                agent_type_enum = AgentType(agent_type)
            except ValueError:
                raise InvalidParams(f"Invalid agent type. Must be one of: {', '.join([t.value for t in AgentType])}")
            
            # Validate autonomy level
            try:
                autonomy_level_enum = AutonomyLevel(autonomy_level)
            except ValueError:
                raise InvalidParams(f"Invalid autonomy level. Must be one of: {', '.join([l.value for l in AutonomyLevel])}")
            
            # Build configuration
            config = get_default_config(agent_type_enum)
            config.autonomy_level = autonomy_level_enum
            config.human_approval_required = human_approval_required
            config.monitoring_interval = timedelta(seconds=monitoring_interval)
            
            # Apply optimization frequency
            frequency_map = {
                "never": timedelta(days=365),
                "hourly": timedelta(hours=1),
                "daily": timedelta(days=1),
                "weekly": timedelta(weeks=1),
                "adaptive": timedelta(hours=4)
            }
            config.optimization_frequency = frequency_map.get(optimization_frequency, timedelta(hours=1))
            
            # Apply resource limits
            if resource_limits:
                config.resource_limits.update(resource_limits)
            else:
                # Default resource limits
                config.resource_limits = {
                    "cpu": 50.0,  # 50% CPU
                    "memory": 1024.0,  # 1GB memory
                    "actions_per_minute": 10.0
                }
            
            # Apply safety constraints
            if safety_constraints:
                config.safety_constraints.update(safety_constraints)
            else:
                # Default safety constraints
                config.safety_constraints = {
                    "max_risk_score": 0.8,
                    "forbidden_actions": [],
                    "require_approval_above_risk": 0.7
                }
            
            # Apply custom configuration
            if agent_config:
                if "decision_threshold" in agent_config:
                    config.decision_threshold = ConfidenceScore(agent_config["decision_threshold"])
                if "risk_tolerance" in agent_config:
                    config.risk_tolerance = RiskScore(agent_config["risk_tolerance"])
                if "learning_rate" in agent_config:
                    config.learning_rate = agent_config["learning_rate"]
                if "max_concurrent_actions" in agent_config:
                    config.max_concurrent_actions = agent_config["max_concurrent_actions"]
            
            if ctx:
                await ctx.report_progress(30, 100, "Creating autonomous agent")
            
            # Create agent
            result = await manager.create_agent(agent_type_enum, config)
            
            if result.is_left():
                error = result.get_left()
                await _log_agent_operation("create", details={"error": str(error)}, success=False)
                raise InternalError(f"Failed to create agent: {error.message}")
            
            agent_id = result.get_right()
            
            if ctx:
                await ctx.report_progress(60, 100, f"Agent created: {agent_id}")
            
            # Add initial goals if provided
            if goals:
                agent = manager.agents[agent_id]
                for goal_data in goals:
                    goal = AgentGoal(
                        goal_id=create_goal_id(),
                        description=goal_data.get("description", "Unnamed goal"),
                        priority=GoalPriority(goal_data.get("priority", "medium")),
                        target_metrics={k: PerformanceMetric(v) for k, v in goal_data.get("target_metrics", {}).items()},
                        success_criteria=goal_data.get("success_criteria", ["Goal completed"]),
                        constraints=goal_data.get("constraints", {}),
                        deadline=datetime.fromisoformat(goal_data["deadline"]) if "deadline" in goal_data else None
                    )
                    
                    goal_result = await agent.add_goal(goal)
                    if goal_result.is_left():
                        if ctx:
                            await ctx.warn(f"Failed to add goal: {goal_result.get_left().message}")
            
            if ctx:
                await ctx.report_progress(100, 100, "Agent created successfully")
            
            await _log_agent_operation("create", agent_id=agent_id, details={
                "agent_type": agent_type,
                "autonomy_level": autonomy_level,
                "goals_count": len(goals) if goals else 0
            })
            
            return {
                "success": True,
                "operation": "create",
                "agent_id": agent_id,
                "agent_type": agent_type,
                "autonomy_level": autonomy_level,
                "status": "created",
                "configuration": {
                    "decision_threshold": config.decision_threshold,
                    "risk_tolerance": config.risk_tolerance,
                    "learning_rate": config.learning_rate,
                    "resource_limits": config.resource_limits,
                    "safety_constraints": config.safety_constraints
                }
            }
        
        # For other operations, we need an agent_id
        if "agent_id" not in agent_config:
            # Try to find the first available agent
            agents = manager.list_agents()
            if not agents:
                raise InvalidParams("No agents available. Create an agent first.")
            agent_id = AgentId(list(agents.keys())[0])
        else:
            agent_id = AgentId(agent_config["agent_id"])
        
        # Handle start operation
        if operation == "start":
            if ctx:
                await ctx.report_progress(50, 100, f"Starting agent {agent_id}")
            
            result = await manager.start_agent(agent_id)
            
            if result.is_left():
                error = result.get_left()
                await _log_agent_operation("start", agent_id=agent_id, details={"error": str(error)}, success=False)
                raise InternalError(f"Failed to start agent: {error.message}")
            
            if ctx:
                await ctx.report_progress(100, 100, "Agent started successfully")
            
            await _log_agent_operation("start", agent_id=agent_id)
            
            return {
                "success": True,
                "operation": "start",
                "agent_id": agent_id,
                "status": "active",
                "message": "Agent started autonomous execution"
            }
        
        # Handle stop operation
        elif operation == "stop":
            if ctx:
                await ctx.report_progress(50, 100, f"Stopping agent {agent_id}")
            
            result = await manager.stop_agent(agent_id)
            
            if result.is_left():
                error = result.get_left()
                await _log_agent_operation("stop", agent_id=agent_id, details={"error": str(error)}, success=False)
                raise InternalError(f"Failed to stop agent: {error.message}")
            
            if ctx:
                await ctx.report_progress(100, 100, "Agent stopped successfully")
            
            await _log_agent_operation("stop", agent_id=agent_id)
            
            return {
                "success": True,
                "operation": "stop",
                "agent_id": agent_id,
                "status": "paused",
                "message": "Agent stopped autonomous execution"
            }
        
        # Handle monitor operation
        elif operation == "monitor":
            if ctx:
                await ctx.report_progress(30, 100, f"Monitoring agent {agent_id}")
            
            # Get agent status
            status_result = manager.get_agent_status(agent_id)
            
            if status_result.is_left():
                error = status_result.get_left()
                await _log_agent_operation("monitor", agent_id=agent_id, details={"error": str(error)}, success=False)
                raise InternalError(f"Failed to get agent status: {error.message}")
            
            status = status_result.get_right()
            
            # Execute a single cycle if agent is active
            agent = manager.agents.get(agent_id)
            cycle_result = None
            if agent and agent.state.status == AgentStatus.ACTIVE:
                if ctx:
                    await ctx.report_progress(60, 100, "Executing monitoring cycle")
                
                cycle_result_either = await agent.execute_single_cycle()
                if cycle_result_either.is_right():
                    cycle_result = cycle_result_either.get_right()
            
            if ctx:
                await ctx.report_progress(100, 100, "Monitoring complete")
            
            await _log_agent_operation("monitor", agent_id=agent_id, details={"status": status["status"]})
            
            return {
                "success": True,
                "operation": "monitor",
                "agent_id": agent_id,
                "status": status,
                "last_cycle": cycle_result,
                "timestamp": datetime.now(UTC).isoformat()
            }
        
        # Handle optimize operation
        elif operation == "optimize":
            if ctx:
                await ctx.report_progress(30, 100, f"Optimizing agent {agent_id}")
            
            agent = manager.agents.get(agent_id)
            if not agent:
                raise InvalidParams(f"Agent {agent_id} not found")
            
            # Perform optimization
            await agent._perform_self_optimization()
            
            if ctx:
                await ctx.report_progress(100, 100, "Optimization complete")
            
            await _log_agent_operation("optimize", agent_id=agent_id)
            
            return {
                "success": True,
                "operation": "optimize",
                "agent_id": agent_id,
                "optimization_cycles": agent.state.metrics.optimization_cycles,
                "current_config": {
                    "decision_threshold": agent.state.configuration.decision_threshold,
                    "risk_tolerance": agent.state.configuration.risk_tolerance,
                    "learning_rate": agent.state.configuration.learning_rate
                },
                "message": "Agent optimization completed"
            }
        
        # Handle configure operation
        elif operation == "configure":
            if ctx:
                await ctx.report_progress(30, 100, f"Configuring agent {agent_id}")
            
            agent = manager.agents.get(agent_id)
            if not agent:
                raise InvalidParams(f"Agent {agent_id} not found")
            
            # Update configuration
            if autonomy_level != agent.state.configuration.autonomy_level.value:
                try:
                    agent.state.configuration.autonomy_level = AutonomyLevel(autonomy_level)
                except ValueError:
                    raise InvalidParams(f"Invalid autonomy level: {autonomy_level}")
            
            if resource_limits:
                agent.state.configuration.resource_limits.update(resource_limits)
            
            if safety_constraints:
                agent.state.configuration.safety_constraints.update(safety_constraints)
            
            agent.state.configuration.human_approval_required = human_approval_required
            agent.state.configuration.monitoring_interval = timedelta(seconds=monitoring_interval)
            
            if ctx:
                await ctx.report_progress(100, 100, "Configuration updated")
            
            await _log_agent_operation("configure", agent_id=agent_id, details={
                "autonomy_level": autonomy_level,
                "human_approval_required": human_approval_required
            })
            
            return {
                "success": True,
                "operation": "configure",
                "agent_id": agent_id,
                "configuration": {
                    "autonomy_level": agent.state.configuration.autonomy_level.value,
                    "human_approval_required": agent.state.configuration.human_approval_required,
                    "monitoring_interval": agent.state.configuration.monitoring_interval.total_seconds(),
                    "resource_limits": agent.state.configuration.resource_limits,
                    "safety_constraints": agent.state.configuration.safety_constraints
                },
                "message": "Agent configuration updated"
            }
        
        # Handle add_goal operation
        elif operation == "add_goal":
            if ctx:
                await ctx.report_progress(30, 100, f"Adding goals to agent {agent_id}")
            
            if not goals:
                raise InvalidParams("No goals provided to add")
            
            agent = manager.agents.get(agent_id)
            if not agent:
                raise InvalidParams(f"Agent {agent_id} not found")
            
            added_goals = []
            for goal_data in goals:
                goal = AgentGoal(
                    goal_id=create_goal_id(),
                    description=goal_data.get("description", "Unnamed goal"),
                    priority=GoalPriority(goal_data.get("priority", "medium")),
                    target_metrics={k: PerformanceMetric(v) for k, v in goal_data.get("target_metrics", {}).items()},
                    success_criteria=goal_data.get("success_criteria", ["Goal completed"]),
                    constraints=goal_data.get("constraints", {}),
                    deadline=datetime.fromisoformat(goal_data["deadline"]) if "deadline" in goal_data else None,
                    resource_requirements=goal_data.get("resource_requirements", {})
                )
                
                result = await agent.add_goal(goal)
                if result.is_right():
                    added_goals.append({
                        "goal_id": goal.goal_id,
                        "description": goal.description,
                        "priority": goal.priority.value,
                        "urgency_score": goal.get_urgency_score()
                    })
                else:
                    if ctx:
                        await ctx.warn(f"Failed to add goal: {result.get_left().message}")
            
            if ctx:
                await ctx.report_progress(100, 100, f"Added {len(added_goals)} goals")
            
            await _log_agent_operation("add_goal", agent_id=agent_id, details={
                "goals_added": len(added_goals)
            })
            
            return {
                "success": True,
                "operation": "add_goal",
                "agent_id": agent_id,
                "goals_added": added_goals,
                "total_active_goals": len(agent.state.current_goals),
                "message": f"Added {len(added_goals)} goals to agent"
            }
        
        # Handle status operation
        elif operation == "status":
            if ctx:
                await ctx.report_progress(50, 100, f"Getting status for agent {agent_id}")
            
            status_result = manager.get_agent_status(agent_id)
            
            if status_result.is_left():
                error = status_result.get_left()
                await _log_agent_operation("status", agent_id=agent_id, details={"error": str(error)}, success=False)
                raise InternalError(f"Failed to get agent status: {error.message}")
            
            status = status_result.get_right()
            
            if ctx:
                await ctx.report_progress(100, 100, "Status retrieved")
            
            return {
                "success": True,
                "operation": "status",
                "agent_status": status
            }
        
        else:
            raise InvalidParams(f"Operation '{operation}' not implemented")
            
    except (InvalidParams, InternalError):
        raise
    except Exception as e:
        logging.error(f"Autonomous agent operation failed: {e}")
        raise InternalError(f"Autonomous agent operation failed: {str(e)}")


# Register the tool
def register_autonomous_agent_tools(server):
    """Register autonomous agent tools with the MCP server."""
    server.add_tool(
        km_autonomous_agent,
        name="km_autonomous_agent",
        description="Create and manage self-managing automation agents with learning capabilities",
        input_schema={
            "type": "object",
            "properties": {
                "operation": {
                    "type": "string",
                    "enum": ["create", "start", "stop", "configure", "monitor", "optimize", "add_goal", "status", "list"],
                    "description": "Operation to perform on autonomous agent"
                },
                "agent_type": {
                    "type": "string",
                    "enum": ["general", "optimizer", "monitor", "learner", "coordinator", "healer", "planner", "resource_manager"],
                    "default": "general",
                    "description": "Type of autonomous agent to create"
                },
                "agent_config": {
                    "type": "object",
                    "description": "Custom agent configuration parameters",
                    "properties": {
                        "agent_id": {"type": "string", "description": "Agent ID for operations"},
                        "decision_threshold": {"type": "number", "minimum": 0, "maximum": 1},
                        "risk_tolerance": {"type": "number", "minimum": 0, "maximum": 1},
                        "learning_rate": {"type": "number", "minimum": 0, "maximum": 1},
                        "max_concurrent_actions": {"type": "integer", "minimum": 1, "maximum": 10}
                    }
                },
                "goals": {
                    "type": "array",
                    "description": "Goals and objectives for the agent",
                    "items": {
                        "type": "object",
                        "properties": {
                            "description": {"type": "string"},
                            "priority": {"type": "string", "enum": ["low", "medium", "high", "critical", "emergency"]},
                            "target_metrics": {"type": "object"},
                            "success_criteria": {"type": "array", "items": {"type": "string"}},
                            "constraints": {"type": "object"},
                            "deadline": {"type": "string", "format": "date-time"},
                            "resource_requirements": {"type": "object"}
                        }
                    }
                },
                "learning_mode": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable learning and adaptation"
                },
                "autonomy_level": {
                    "type": "string",
                    "enum": ["manual", "supervised", "autonomous", "full"],
                    "default": "supervised",
                    "description": "Level of autonomy and human oversight"
                },
                "resource_limits": {
                    "type": "object",
                    "description": "Resource usage limits (CPU, memory, actions)",
                    "properties": {
                        "cpu": {"type": "number", "minimum": 0, "maximum": 100},
                        "memory": {"type": "number", "minimum": 0},
                        "actions_per_minute": {"type": "number", "minimum": 0}
                    }
                },
                "safety_constraints": {
                    "type": "object",
                    "description": "Safety rules and constraints",
                    "properties": {
                        "max_risk_score": {"type": "number", "minimum": 0, "maximum": 1},
                        "forbidden_actions": {"type": "array", "items": {"type": "string"}},
                        "require_approval_above_risk": {"type": "number", "minimum": 0, "maximum": 1}
                    }
                },
                "communication_enabled": {
                    "type": "boolean",
                    "default": True,
                    "description": "Enable inter-agent communication"
                },
                "monitoring_interval": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 3600,
                    "default": 60,
                    "description": "Self-monitoring interval in seconds"
                },
                "optimization_frequency": {
                    "type": "string",
                    "enum": ["never", "hourly", "daily", "weekly", "adaptive"],
                    "default": "hourly",
                    "description": "How often agent optimizes itself"
                },
                "human_approval_required": {
                    "type": "boolean",
                    "default": False,
                    "description": "Require human approval for actions"
                },
                "timeout": {
                    "type": "integer",
                    "minimum": 1,
                    "maximum": 600,
                    "default": 300,
                    "description": "Operation timeout in seconds"
                }
            },
            "required": ["operation"]
        }
    )
    
    return server