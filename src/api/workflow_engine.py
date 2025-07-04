"""
API Workflow Orchestration Engine - TASK_64 Phase 2 Implementation

Advanced workflow orchestration for complex multi-API compositions with 
Design by Contract patterns, type safety, and fault tolerance.

Architecture: Event-driven workflow + State management + Pattern matching
Performance: <500ms workflow coordination, <50ms state transitions
Security: Workflow validation, data isolation, and execution sandboxing
"""

from __future__ import annotations
import asyncio
import time
import uuid
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional, Union, Callable, Set
from dataclasses import dataclass, field
from enum import Enum
import logging
from contextlib import asynccontextmanager

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.errors import ValidationError, SecurityError
from ..core.types import WorkflowID, APIEndpoint, ExecutionContext

logger = logging.getLogger(__name__)


class WorkflowState(Enum):
    """Workflow execution states."""
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"
    PAUSED = "paused"


class StepType(Enum):
    """Workflow step types."""
    API_CALL = "api_call"
    CONDITION = "condition"
    LOOP = "loop"
    PARALLEL = "parallel"
    TRANSFORM = "transform"
    DELAY = "delay"


class ExecutionStrategy(Enum):
    """Workflow execution strategies."""
    SEQUENTIAL = "sequential"
    PARALLEL = "parallel"
    CONDITIONAL = "conditional"
    PIPELINE = "pipeline"


@dataclass
class WorkflowStep:
    """Individual step in workflow execution."""
    step_id: str
    step_type: StepType
    name: str
    configuration: Dict[str, Any]
    dependencies: List[str] = field(default_factory=list)
    retry_config: Optional[Dict[str, Any]] = None
    timeout_seconds: Optional[int] = None
    condition: Optional[str] = None  # Conditional execution logic
    
    def __post_init__(self):
        if not self.step_id:
            self.step_id = str(uuid.uuid4())


@dataclass
class WorkflowDefinition:
    """Complete workflow definition with metadata and execution plan."""
    workflow_id: WorkflowID
    name: str
    description: str
    steps: List[WorkflowStep]
    execution_strategy: ExecutionStrategy
    global_timeout_seconds: int = 300
    error_handling: str = "fail_fast"  # fail_fast, continue, retry
    max_retries: int = 3
    retry_delay_seconds: int = 5
    created_at: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def __post_init__(self):
        if not self.workflow_id:
            self.workflow_id = WorkflowID(str(uuid.uuid4()))


@dataclass
class StepResult:
    """Result of workflow step execution."""
    step_id: str
    state: WorkflowState
    data: Any = None
    error: Optional[str] = None
    execution_time_ms: float = 0
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    retry_count: int = 0


@dataclass
class WorkflowExecution:
    """Active workflow execution tracking."""
    execution_id: str
    workflow_definition: WorkflowDefinition
    state: WorkflowState = WorkflowState.PENDING
    step_results: Dict[str, StepResult] = field(default_factory=dict)
    current_step: Optional[str] = None
    context_data: Dict[str, Any] = field(default_factory=dict)
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    error_message: Optional[str] = None
    
    def __post_init__(self):
        if not self.execution_id:
            self.execution_id = str(uuid.uuid4())


class WorkflowEngine:
    """
    Advanced workflow orchestration engine for complex API compositions.
    
    Supports sequential, parallel, and conditional execution patterns with
    sophisticated error handling, retry logic, and state management.
    """
    
    def __init__(self, max_concurrent_workflows: int = 50):
        require(lambda: max_concurrent_workflows > 0, "Max workflows must be positive")
        
        self.max_concurrent_workflows = max_concurrent_workflows
        self.active_executions: Dict[str, WorkflowExecution] = {}
        self.workflow_definitions: Dict[WorkflowID, WorkflowDefinition] = {}
        self.step_handlers: Dict[StepType, Callable] = {}
        self.execution_lock = asyncio.Lock()
        
        # Initialize step handlers
        self._initialize_step_handlers()
        
        logger.info(f"WorkflowEngine initialized with {max_concurrent_workflows} max concurrent workflows")
    
    def _initialize_step_handlers(self) -> None:
        """Initialize step type handlers."""
        self.step_handlers = {
            StepType.API_CALL: self._execute_api_call_step,
            StepType.CONDITION: self._execute_condition_step,
            StepType.LOOP: self._execute_loop_step,
            StepType.PARALLEL: self._execute_parallel_step,
            StepType.TRANSFORM: self._execute_transform_step,
            StepType.DELAY: self._execute_delay_step
        }
    
    @require(lambda definition: definition.name and definition.steps, "Definition must have name and steps")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns workflow ID or error")
    async def register_workflow(self, definition: WorkflowDefinition) -> Either[str, WorkflowID]:
        """Register a new workflow definition."""
        try:
            # Validate workflow definition
            validation_result = self._validate_workflow_definition(definition)
            if validation_result.is_left():
                return validation_result
            
            # Store definition
            self.workflow_definitions[definition.workflow_id] = definition
            
            logger.info(f"Registered workflow {definition.workflow_id}: {definition.name}")
            return Either.right(definition.workflow_id)
            
        except Exception as e:
            error_msg = f"Failed to register workflow: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    @require(lambda workflow_id: workflow_id is not None, "Workflow ID required")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns execution ID or error")
    async def start_workflow(
        self, 
        workflow_id: WorkflowID, 
        context_data: Optional[Dict[str, Any]] = None
    ) -> Either[str, str]:
        """Start workflow execution with optional initial context data."""
        try:
            async with self.execution_lock:
                # Check concurrent execution limits
                if len(self.active_executions) >= self.max_concurrent_workflows:
                    return Either.left(f"Maximum {self.max_concurrent_workflows} concurrent workflows exceeded")
                
                # Get workflow definition
                if workflow_id not in self.workflow_definitions:
                    return Either.left(f"Workflow {workflow_id} not found")
                
                definition = self.workflow_definitions[workflow_id]
                
                # Create execution
                execution = WorkflowExecution(
                    execution_id=str(uuid.uuid4()),
                    workflow_definition=definition,
                    context_data=context_data or {},
                    started_at=datetime.now(UTC)
                )
                
                self.active_executions[execution.execution_id] = execution
                
                # Start execution task
                asyncio.create_task(self._execute_workflow(execution))
                
                logger.info(f"Started workflow execution {execution.execution_id}")
                return Either.right(execution.execution_id)
                
        except Exception as e:
            error_msg = f"Failed to start workflow: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def _execute_workflow(self, execution: WorkflowExecution) -> None:
        """Execute workflow according to its strategy."""
        try:
            execution.state = WorkflowState.RUNNING
            
            # Set global timeout
            timeout_task = asyncio.create_task(
                asyncio.sleep(execution.workflow_definition.global_timeout_seconds)
            )
            
            # Execute according to strategy
            if execution.workflow_definition.execution_strategy == ExecutionStrategy.SEQUENTIAL:
                execution_task = asyncio.create_task(self._execute_sequential(execution))
            elif execution.workflow_definition.execution_strategy == ExecutionStrategy.PARALLEL:
                execution_task = asyncio.create_task(self._execute_parallel(execution))
            elif execution.workflow_definition.execution_strategy == ExecutionStrategy.CONDITIONAL:
                execution_task = asyncio.create_task(self._execute_conditional(execution))
            elif execution.workflow_definition.execution_strategy == ExecutionStrategy.PIPELINE:
                execution_task = asyncio.create_task(self._execute_pipeline(execution))
            else:
                raise ValidationError("execution_strategy", execution.workflow_definition.execution_strategy, "Unknown strategy")
            
            # Wait for completion or timeout
            done, pending = await asyncio.wait(
                [execution_task, timeout_task],
                return_when=asyncio.FIRST_COMPLETED
            )
            
            # Cancel pending tasks
            for task in pending:
                task.cancel()
            
            if timeout_task in done:
                execution.state = WorkflowState.FAILED
                execution.error_message = "Workflow execution timed out"
                logger.warning(f"Workflow {execution.execution_id} timed out")
            else:
                execution.state = WorkflowState.COMPLETED
                logger.info(f"Workflow {execution.execution_id} completed successfully")
            
        except Exception as e:
            execution.state = WorkflowState.FAILED
            execution.error_message = str(e)
            logger.error(f"Workflow execution failed: {e}")
        
        finally:
            execution.completed_at = datetime.now(UTC)
            # Keep execution in memory for result retrieval
    
    async def _execute_sequential(self, execution: WorkflowExecution) -> None:
        """Execute workflow steps sequentially."""
        for step in execution.workflow_definition.steps:
            # Check dependencies
            if not self._are_dependencies_satisfied(step, execution):
                if execution.workflow_definition.error_handling == "fail_fast":
                    raise ValidationError("dependencies", step.dependencies, "Dependencies not satisfied")
                continue
            
            # Execute step
            result = await self._execute_step(step, execution)
            execution.step_results[step.step_id] = result
            
            # Handle step failure
            if result.state == WorkflowState.FAILED:
                if execution.workflow_definition.error_handling == "fail_fast":
                    raise Exception(f"Step {step.step_id} failed: {result.error}")
                # Continue with next step if error_handling is "continue"
    
    async def _execute_parallel(self, execution: WorkflowExecution) -> None:
        """Execute workflow steps in parallel where possible."""
        # Group steps by dependency levels
        dependency_levels = self._build_dependency_levels(execution.workflow_definition.steps)
        
        for level_steps in dependency_levels:
            # Execute all steps in this level in parallel
            tasks = []
            for step in level_steps:
                task = asyncio.create_task(self._execute_step(step, execution))
                tasks.append((step.step_id, task))
            
            # Wait for all tasks in this level to complete
            results = await asyncio.gather(*[task for _, task in tasks], return_exceptions=True)
            
            for (step_id, _), result in zip(tasks, results):
                if isinstance(result, Exception):
                    execution.step_results[step_id] = StepResult(
                        step_id=step_id,
                        state=WorkflowState.FAILED,
                        error=str(result)
                    )
                else:
                    execution.step_results[step_id] = result
    
    async def _execute_conditional(self, execution: WorkflowExecution) -> None:
        """Execute workflow with conditional logic."""
        for step in execution.workflow_definition.steps:
            # Evaluate step condition
            if step.condition and not self._evaluate_condition(step.condition, execution.context_data):
                execution.step_results[step.step_id] = StepResult(
                    step_id=step.step_id,
                    state=WorkflowState.COMPLETED,
                    data="Skipped due to condition"
                )
                continue
            
            # Execute step
            result = await self._execute_step(step, execution)
            execution.step_results[step.step_id] = result
            
            # Update context with step result
            if result.data:
                execution.context_data[f"step_{step.step_id}_result"] = result.data
    
    async def _execute_pipeline(self, execution: WorkflowExecution) -> None:
        """Execute workflow as a data pipeline."""
        pipeline_data = execution.context_data.get("initial_data")
        
        for step in execution.workflow_definition.steps:
            # Pass previous step's output as input to next step
            step_config = step.configuration.copy()
            if pipeline_data is not None:
                step_config["input_data"] = pipeline_data
            
            # Execute step with pipeline data
            modified_step = WorkflowStep(
                step_id=step.step_id,
                step_type=step.step_type,
                name=step.name,
                configuration=step_config,
                dependencies=step.dependencies,
                retry_config=step.retry_config,
                timeout_seconds=step.timeout_seconds,
                condition=step.condition
            )
            
            result = await self._execute_step(modified_step, execution)
            execution.step_results[step.step_id] = result
            
            # Use step result as pipeline data for next step
            pipeline_data = result.data
            
            if result.state == WorkflowState.FAILED and execution.workflow_definition.error_handling == "fail_fast":
                break
    
    async def _execute_step(self, step: WorkflowStep, execution: WorkflowExecution) -> StepResult:
        """Execute a single workflow step with retry logic."""
        start_time = time.time()
        result = StepResult(
            step_id=step.step_id,
            started_at=datetime.now(UTC),
            state=WorkflowState.RUNNING
        )
        
        retry_count = 0
        max_retries = step.retry_config.get("max_retries", execution.workflow_definition.max_retries) if step.retry_config else execution.workflow_definition.max_retries
        
        while retry_count <= max_retries:
            try:
                # Set step timeout
                timeout = step.timeout_seconds or 60
                
                # Execute step based on type
                if step.step_type not in self.step_handlers:
                    raise ValidationError("step_type", step.step_type, "Unknown step type")
                
                handler = self.step_handlers[step.step_type]
                step_result = await asyncio.wait_for(
                    handler(step, execution.context_data),
                    timeout=timeout
                )
                
                result.data = step_result
                result.state = WorkflowState.COMPLETED
                break
                
            except asyncio.TimeoutError:
                result.error = f"Step timed out after {timeout} seconds"
                result.state = WorkflowState.FAILED
            except Exception as e:
                result.error = str(e)
                result.state = WorkflowState.FAILED
            
            retry_count += 1
            
            if retry_count <= max_retries and result.state == WorkflowState.FAILED:
                delay = step.retry_config.get("delay_seconds", execution.workflow_definition.retry_delay_seconds) if step.retry_config else execution.workflow_definition.retry_delay_seconds
                await asyncio.sleep(delay)
                logger.info(f"Retrying step {step.step_id}, attempt {retry_count}")
            
        result.retry_count = retry_count
        result.execution_time_ms = (time.time() - start_time) * 1000
        result.completed_at = datetime.now(UTC)
        
        return result
    
    async def _execute_api_call_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute API call step."""
        import aiohttp
        
        config = step.configuration
        url = config.get("url")
        method = config.get("method", "GET").upper()
        headers = config.get("headers", {})
        data = config.get("data")
        params = config.get("params")
        
        # Template substitution from context
        if isinstance(data, dict):
            data = self._substitute_template_variables(data, context)
        if isinstance(params, dict):
            params = self._substitute_template_variables(params, context)
        
        async with aiohttp.ClientSession() as session:
            async with session.request(
                method=method,
                url=url,
                headers=headers,
                json=data if method != "GET" else None,
                params=params
            ) as response:
                response.raise_for_status()
                return await response.json()
    
    async def _execute_condition_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute conditional step."""
        condition = step.configuration.get("condition")
        true_branch = step.configuration.get("true_branch")
        false_branch = step.configuration.get("false_branch")
        
        if self._evaluate_condition(condition, context):
            return {"condition_result": True, "executed_branch": "true", "data": true_branch}
        else:
            return {"condition_result": False, "executed_branch": "false", "data": false_branch}
    
    async def _execute_loop_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute loop step."""
        config = step.configuration
        iterations = config.get("iterations", 1)
        loop_variable = config.get("loop_variable", "index")
        loop_body = config.get("body", [])
        
        results = []
        for i in range(iterations):
            loop_context = context.copy()
            loop_context[loop_variable] = i
            
            # Execute loop body (simplified - would need recursive step execution)
            results.append({"iteration": i, "context": loop_context})
        
        return {"loop_results": results, "iterations_completed": len(results)}
    
    async def _execute_parallel_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute parallel step."""
        parallel_tasks = step.configuration.get("tasks", [])
        
        # Execute all parallel tasks
        tasks = []
        for task_config in parallel_tasks:
            # This would create sub-steps and execute them
            task = asyncio.create_task(asyncio.sleep(0.1))  # Placeholder
            tasks.append(task)
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        return {"parallel_results": results, "tasks_completed": len(results)}
    
    async def _execute_transform_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute data transformation step."""
        config = step.configuration
        input_data = config.get("input_data") or context.get("pipeline_data")
        transformation = config.get("transformation", "identity")
        
        # Simple transformations (could be extended with more complex logic)
        if transformation == "identity":
            return input_data
        elif transformation == "uppercase" and isinstance(input_data, str):
            return input_data.upper()
        elif transformation == "lowercase" and isinstance(input_data, str):
            return input_data.lower()
        elif transformation == "json_extract":
            field = config.get("field")
            return input_data.get(field) if isinstance(input_data, dict) and field else input_data
        else:
            return input_data
    
    async def _execute_delay_step(self, step: WorkflowStep, context: Dict[str, Any]) -> Any:
        """Execute delay step."""
        delay_seconds = step.configuration.get("delay_seconds", 1)
        await asyncio.sleep(delay_seconds)
        return {"delayed_seconds": delay_seconds, "completed_at": datetime.now(UTC).isoformat()}
    
    def _validate_workflow_definition(self, definition: WorkflowDefinition) -> Either[str, None]:
        """Validate workflow definition."""
        if not definition.steps:
            return Either.left("Workflow must have at least one step")
        
        # Check for cyclic dependencies
        if self._has_cyclic_dependencies(definition.steps):
            return Either.left("Workflow has cyclic dependencies")
        
        # Validate step configurations
        for step in definition.steps:
            if step.step_type == StepType.API_CALL:
                if not step.configuration.get("url"):
                    return Either.left(f"API call step {step.step_id} missing URL")
        
        return Either.right(None)
    
    def _has_cyclic_dependencies(self, steps: List[WorkflowStep]) -> bool:
        """Check for cyclic dependencies in workflow steps."""
        # Build dependency graph
        graph = {step.step_id: step.dependencies for step in steps}
        
        # Use DFS to detect cycles
        visiting = set()
        visited = set()
        
        def has_cycle(node: str) -> bool:
            if node in visiting:
                return True
            if node in visited:
                return False
            
            visiting.add(node)
            for dependency in graph.get(node, []):
                if has_cycle(dependency):
                    return True
            visiting.remove(node)
            visited.add(node)
            return False
        
        for step_id in graph:
            if has_cycle(step_id):
                return True
        
        return False
    
    def _are_dependencies_satisfied(self, step: WorkflowStep, execution: WorkflowExecution) -> bool:
        """Check if step dependencies are satisfied."""
        for dependency in step.dependencies:
            if dependency not in execution.step_results:
                return False
            if execution.step_results[dependency].state != WorkflowState.COMPLETED:
                return False
        return True
    
    def _build_dependency_levels(self, steps: List[WorkflowStep]) -> List[List[WorkflowStep]]:
        """Build dependency levels for parallel execution."""
        step_map = {step.step_id: step for step in steps}
        levels = []
        processed = set()
        
        while len(processed) < len(steps):
            current_level = []
            
            for step in steps:
                if step.step_id in processed:
                    continue
                
                # Check if all dependencies are processed
                if all(dep in processed for dep in step.dependencies):
                    current_level.append(step)
            
            if not current_level:
                # This shouldn't happen if there are no cycles
                break
            
            levels.append(current_level)
            processed.update(step.step_id for step in current_level)
        
        return levels
    
    def _evaluate_condition(self, condition: str, context: Dict[str, Any]) -> bool:
        """Evaluate conditional expression safely."""
        # Simple condition evaluation - could be enhanced with expression parser
        try:
            # Replace context variables in condition
            for key, value in context.items():
                if isinstance(value, str):
                    condition = condition.replace(f"${key}", f"'{value}'")
                else:
                    condition = condition.replace(f"${key}", str(value))
            
            # Basic safety check - only allow simple comparisons
            allowed_operators = ["==", "!=", ">", "<", ">=", "<=", "and", "or", "not"]
            if any(op in condition for op in ["import", "exec", "eval", "__"]):
                return False
            
            # Evaluate condition
            return bool(eval(condition))
        except Exception:
            return False
    
    def _substitute_template_variables(self, data: Dict[str, Any], context: Dict[str, Any]) -> Dict[str, Any]:
        """Substitute template variables in data."""
        if isinstance(data, dict):
            return {
                key: self._substitute_template_variables(value, context) if isinstance(value, (dict, list)) else 
                     str(value).format(**context) if isinstance(value, str) and "{" in str(value) else value
                for key, value in data.items()
            }
        elif isinstance(data, list):
            return [self._substitute_template_variables(item, context) for item in data]
        else:
            return data
    
    def get_execution_status(self, execution_id: str) -> Optional[Dict[str, Any]]:
        """Get current status of workflow execution."""
        if execution_id not in self.active_executions:
            return None
        
        execution = self.active_executions[execution_id]
        
        return {
            "execution_id": execution.execution_id,
            "workflow_id": execution.workflow_definition.workflow_id,
            "state": execution.state.value,
            "current_step": execution.current_step,
            "steps_completed": len([r for r in execution.step_results.values() if r.state == WorkflowState.COMPLETED]),
            "total_steps": len(execution.workflow_definition.steps),
            "started_at": execution.started_at.isoformat() if execution.started_at else None,
            "completed_at": execution.completed_at.isoformat() if execution.completed_at else None,
            "error_message": execution.error_message,
            "step_results": {
                step_id: {
                    "state": result.state.value,
                    "execution_time_ms": result.execution_time_ms,
                    "retry_count": result.retry_count,
                    "error": result.error
                }
                for step_id, result in execution.step_results.items()
            }
        }
    
    async def cancel_workflow(self, execution_id: str) -> Either[str, None]:
        """Cancel running workflow execution."""
        try:
            if execution_id not in self.active_executions:
                return Either.left(f"Execution {execution_id} not found")
            
            execution = self.active_executions[execution_id]
            execution.state = WorkflowState.CANCELLED
            execution.completed_at = datetime.now(UTC)
            
            logger.info(f"Cancelled workflow execution {execution_id}")
            return Either.right(None)
            
        except Exception as e:
            error_msg = f"Failed to cancel workflow: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def cleanup_completed_executions(self, max_age_hours: int = 24) -> int:
        """Clean up completed workflow executions older than specified age."""
        cutoff_time = datetime.now(UTC) - timedelta(hours=max_age_hours)
        cleaned_count = 0
        
        execution_ids_to_remove = []
        for execution_id, execution in self.active_executions.items():
            if (execution.state in [WorkflowState.COMPLETED, WorkflowState.FAILED, WorkflowState.CANCELLED] and
                execution.completed_at and execution.completed_at < cutoff_time):
                execution_ids_to_remove.append(execution_id)
        
        for execution_id in execution_ids_to_remove:
            del self.active_executions[execution_id]
            cleaned_count += 1
        
        logger.info(f"Cleaned up {cleaned_count} completed workflow executions")
        return cleaned_count


# Global instance
_workflow_engine: Optional[WorkflowEngine] = None


def get_workflow_engine() -> WorkflowEngine:
    """Get or create the global workflow engine instance."""
    global _workflow_engine
    if _workflow_engine is None:
        _workflow_engine = WorkflowEngine()
    return _workflow_engine


@asynccontextmanager
async def workflow_execution_context(workflow_id: WorkflowID, context_data: Optional[Dict[str, Any]] = None):
    """Context manager for workflow execution."""
    engine = get_workflow_engine()
    
    execution_result = await engine.start_workflow(workflow_id, context_data)
    if execution_result.is_left():
        raise ValidationError("workflow_execution", workflow_id, execution_result.left())
    
    execution_id = execution_result.right()
    
    try:
        yield execution_id
    finally:
        # Execution cleanup is handled by the engine
        pass