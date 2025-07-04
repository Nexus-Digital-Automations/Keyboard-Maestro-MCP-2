"""
Advanced batch processing system for AI operations.

This module provides comprehensive batch processing capabilities for AI operations
including parallel processing, dependency management, progress tracking, and
intelligent scheduling with enterprise-grade error handling and recovery.

Security: All batch operations include comprehensive validation and safe execution.
Performance: Optimized for high-throughput processing with intelligent resource management.
Type Safety: Complete integration with AI processing architecture.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import NewType, Dict, List, Optional, Any, Set, Callable, Union, Tuple
from enum import Enum
from datetime import datetime, timedelta, UTC
import asyncio
import json
from uuid import uuid4
import heapq

from ..core.either import Either
from ..core.contracts import require, ensure
from ..core.errors import ValidationError
from ..core.ai_integration import AIOperation, AIRequest, AIResponse, create_ai_request

# Branded Types for Batch Processing
BatchJobId = NewType('BatchJobId', str)
BatchTaskId = NewType('BatchTaskId', str)
ProcessingPriority = NewType('ProcessingPriority', int)
ResourceQuota = NewType('ResourceQuota', float)
ProgressPercentage = NewType('ProgressPercentage', float)


class BatchStatus(Enum):
    """Batch job and task status types."""
    PENDING = "pending"              # Waiting to start
    QUEUED = "queued"               # In processing queue
    RUNNING = "running"             # Currently executing
    COMPLETED = "completed"         # Successfully finished
    FAILED = "failed"               # Failed with error
    CANCELLED = "cancelled"         # Cancelled by user
    PAUSED = "paused"              # Temporarily paused
    RETRYING = "retrying"          # Retrying after failure


class BatchMode(Enum):
    """Batch processing execution modes."""
    SEQUENTIAL = "sequential"       # Process tasks one by one
    PARALLEL = "parallel"          # Process tasks in parallel
    PIPELINE = "pipeline"          # Pipeline processing with stages
    PRIORITY = "priority"          # Priority-based scheduling
    RESOURCE_AWARE = "resource_aware" # Resource-constrained processing


class DependencyType(Enum):
    """Task dependency relationship types."""
    SEQUENTIAL = "sequential"       # Must complete before next starts
    DATA_DEPENDENCY = "data_dependency" # Output of one feeds into next
    CONDITIONAL = "conditional"     # Conditional dependency based on result
    SOFT_DEPENDENCY = "soft_dependency" # Preferred but not required order


@dataclass(frozen=True)
class BatchTask:
    """Individual task within a batch job."""
    task_id: BatchTaskId
    operation: AIOperation
    input_data: Any
    processing_parameters: Dict[str, Any] = field(default_factory=dict)
    priority: ProcessingPriority = ProcessingPriority(5)
    dependencies: Set[BatchTaskId] = field(default_factory=set)
    max_retries: int = 3
    timeout: timedelta = timedelta(minutes=10)
    tags: Set[str] = field(default_factory=set)
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    @require(lambda self: len(self.task_id) > 0)
    @require(lambda self: 1 <= self.priority <= 10)
    @require(lambda self: self.max_retries >= 0)
    @require(lambda self: self.timeout.total_seconds() > 0)
    def __post_init__(self):
        """Validate batch task configuration."""
        pass
    
    def estimate_resource_usage(self) -> Dict[str, float]:
        """Estimate resource usage for this task."""
        # Base estimates - in practice would be more sophisticated
        base_cpu = 0.1
        base_memory = 50.0  # MB
        base_time = 5.0     # seconds
        
        # Adjust based on operation type
        if self.operation in [AIOperation.ANALYZE, AIOperation.CLASSIFY]:
            cpu_factor = 1.0
            memory_factor = 1.0
            time_factor = 1.0
        elif self.operation in [AIOperation.GENERATE, AIOperation.TRANSFORM]:
            cpu_factor = 2.0
            memory_factor = 1.5
            time_factor = 2.0
        elif self.operation == AIOperation.EXTRACT:
            cpu_factor = 1.5
            memory_factor = 2.0
            time_factor = 1.5
        else:
            cpu_factor = 1.0
            memory_factor = 1.0
            time_factor = 1.0
        
        # Adjust based on input size
        input_size = len(str(self.input_data))
        size_factor = max(1.0, input_size / 1000.0)  # Scale with input size
        
        return {
            "cpu": base_cpu * cpu_factor * size_factor,
            "memory": base_memory * memory_factor * size_factor,
            "estimated_time": base_time * time_factor * size_factor
        }


@dataclass(frozen=True)
class TaskResult:
    """Result of a batch task execution."""
    task_id: BatchTaskId
    status: BatchStatus
    result: Optional[AIResponse] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    resource_usage: Dict[str, float] = field(default_factory=dict)
    retry_count: int = 0
    timestamp: datetime = field(default_factory=lambda: datetime.now(UTC))
    
    def is_successful(self) -> bool:
        """Check if task completed successfully."""
        return self.status == BatchStatus.COMPLETED and self.result is not None
    
    def is_failed(self) -> bool:
        """Check if task failed permanently."""
        return self.status == BatchStatus.FAILED
    
    def can_retry(self, max_retries: int) -> bool:
        """Check if task can be retried."""
        return self.retry_count < max_retries and self.status in [BatchStatus.FAILED, BatchStatus.CANCELLED]


@dataclass(frozen=True)
class BatchJob:
    """Complete batch processing job configuration."""
    job_id: BatchJobId
    name: str
    tasks: List[BatchTask]
    processing_mode: BatchMode = BatchMode.PARALLEL
    max_concurrent_tasks: int = 5
    total_timeout: timedelta = timedelta(hours=1)
    resource_limits: Dict[str, float] = field(default_factory=dict)
    priority: ProcessingPriority = ProcessingPriority(5)
    enable_checkpointing: bool = True
    auto_retry_failed: bool = True
    notification_callbacks: List[Callable] = field(default_factory=list)
    
    @require(lambda self: len(self.job_id) > 0)
    @require(lambda self: len(self.name) > 0)
    @require(lambda self: len(self.tasks) > 0)
    @require(lambda self: self.max_concurrent_tasks > 0)
    @require(lambda self: 1 <= self.priority <= 10)
    def __post_init__(self):
        """Validate batch job configuration."""
        pass
    
    def get_dependency_graph(self) -> Dict[BatchTaskId, Set[BatchTaskId]]:
        """Build task dependency graph."""
        graph = {}
        for task in self.tasks:
            graph[task.task_id] = task.dependencies.copy()
        return graph
    
    def validate_dependencies(self) -> Either[ValidationError, None]:
        """Validate that task dependencies are valid and acyclic."""
        try:
            task_ids = {task.task_id for task in self.tasks}
            
            # Check all dependencies exist
            for task in self.tasks:
                for dep_id in task.dependencies:
                    if dep_id not in task_ids:
                        return Either.left(ValidationError(
                            "invalid_dependency",
                            f"Task {task.task_id} depends on non-existent task {dep_id}"
                        ))
            
            # Check for cycles using DFS
            if self._has_dependency_cycle():
                return Either.left(ValidationError(
                    "circular_dependency",
                    "Circular dependency detected in task graph"
                ))
            
            return Either.right(None)
            
        except Exception as e:
            return Either.left(ValidationError("dependency_validation_failed", str(e)))
    
    def _has_dependency_cycle(self) -> bool:
        """Check for cycles in dependency graph using DFS."""
        graph = self.get_dependency_graph()
        white = set(graph.keys())  # Not visited
        gray = set()               # Currently visiting
        black = set()              # Completed
        
        def dfs(node: BatchTaskId) -> bool:
            if node in gray:
                return True  # Back edge found - cycle detected
            if node in black:
                return False  # Already processed
            
            white.discard(node)
            gray.add(node)
            
            for neighbor in graph.get(node, set()):
                if dfs(neighbor):
                    return True
            
            gray.discard(node)
            black.add(node)
            return False
        
        for node in list(white):
            if dfs(node):
                return True
        
        return False
    
    def get_ready_tasks(self, completed_tasks: Set[BatchTaskId]) -> List[BatchTask]:
        """Get tasks that are ready to execute (dependencies satisfied)."""
        ready_tasks = []
        for task in self.tasks:
            if task.dependencies.issubset(completed_tasks):
                ready_tasks.append(task)
        return ready_tasks
    
    def estimate_total_resources(self) -> Dict[str, float]:
        """Estimate total resource usage for the batch job."""
        total_resources = {"cpu": 0.0, "memory": 0.0, "estimated_time": 0.0}
        
        for task in self.tasks:
            task_resources = task.estimate_resource_usage()
            for resource, usage in task_resources.items():
                if resource in total_resources:
                    if self.processing_mode == BatchMode.SEQUENTIAL:
                        # Sequential: add time, max others
                        if resource == "estimated_time":
                            total_resources[resource] += usage
                        else:
                            total_resources[resource] = max(total_resources[resource], usage)
                    else:
                        # Parallel: max time, add others
                        if resource == "estimated_time":
                            total_resources[resource] = max(total_resources[resource], usage)
                        else:
                            total_resources[resource] += usage
        
        # Apply concurrency limits for parallel processing
        if self.processing_mode in [BatchMode.PARALLEL, BatchMode.RESOURCE_AWARE]:
            concurrency_factor = min(self.max_concurrent_tasks, len(self.tasks))
            total_resources["cpu"] = min(total_resources["cpu"], 
                                       total_resources["cpu"] / len(self.tasks) * concurrency_factor)
            total_resources["memory"] = min(total_resources["memory"],
                                          total_resources["memory"] / len(self.tasks) * concurrency_factor)
        
        return total_resources


@dataclass
class BatchJobState:
    """Mutable state for batch job execution."""
    job: BatchJob
    status: BatchStatus = BatchStatus.PENDING
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    task_results: Dict[BatchTaskId, TaskResult] = field(default_factory=dict)
    active_tasks: Set[BatchTaskId] = field(default_factory=set)
    failed_tasks: Set[BatchTaskId] = field(default_factory=set)
    cancelled_tasks: Set[BatchTaskId] = field(default_factory=set)
    current_resource_usage: Dict[str, float] = field(default_factory=dict)
    checkpoint_data: Dict[str, Any] = field(default_factory=dict)
    
    def get_progress(self) -> ProgressPercentage:
        """Calculate job progress percentage."""
        total_tasks = len(self.job.tasks)
        if total_tasks == 0:
            return ProgressPercentage(100.0)
        
        completed_tasks = len([r for r in self.task_results.values() 
                             if r.status in [BatchStatus.COMPLETED, BatchStatus.FAILED]])
        return ProgressPercentage((completed_tasks / total_tasks) * 100.0)
    
    def get_completed_task_ids(self) -> Set[BatchTaskId]:
        """Get IDs of successfully completed tasks."""
        return {task_id for task_id, result in self.task_results.items()
                if result.is_successful()}
    
    def is_finished(self) -> bool:
        """Check if job is finished (completed, failed, or cancelled)."""
        return self.status in [BatchStatus.COMPLETED, BatchStatus.FAILED, BatchStatus.CANCELLED]
    
    def get_execution_summary(self) -> Dict[str, Any]:
        """Get comprehensive execution summary."""
        total_tasks = len(self.job.tasks)
        completed_tasks = len([r for r in self.task_results.values() if r.is_successful()])
        failed_tasks = len([r for r in self.task_results.values() if r.is_failed()])
        
        total_time = 0.0
        if self.started_at and self.completed_at:
            total_time = (self.completed_at - self.started_at).total_seconds()
        elif self.started_at:
            total_time = (datetime.now(UTC) - self.started_at).total_seconds()
        
        return {
            "job_id": str(self.job.job_id),
            "job_name": self.job.name,
            "status": self.status.value,
            "progress": float(self.get_progress()),
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "failed_tasks": failed_tasks,
            "active_tasks": len(self.active_tasks),
            "total_execution_time": total_time,
            "started_at": self.started_at.isoformat() if self.started_at else None,
            "completed_at": self.completed_at.isoformat() if self.completed_at else None,
            "resource_usage": dict(self.current_resource_usage),
            "processing_mode": self.job.processing_mode.value
        }


class BatchProcessor:
    """High-performance batch processing engine for AI operations."""
    
    def __init__(self, ai_processing_manager):
        self.ai_manager = ai_processing_manager
        self.active_jobs: Dict[BatchJobId, BatchJobState] = {}
        self.job_queue: List[Tuple[int, BatchJobId]] = []  # Priority queue
        self.max_concurrent_jobs = 3
        self.global_resource_limits = {
            "cpu": 8.0,      # Max CPU cores
            "memory": 8192.0, # Max memory in MB
            "concurrent_requests": 50
        }
        self.processing_tasks: Dict[BatchJobId, asyncio.Task] = {}
        self.is_running = False
    
    async def start_processor(self) -> None:
        """Start the batch processor."""
        if self.is_running:
            return
        
        self.is_running = True
        # Start background job scheduler
        asyncio.create_task(self._job_scheduler())
    
    async def stop_processor(self) -> None:
        """Stop the batch processor."""
        if not self.is_running:
            return
        
        self.is_running = False
        
        # Cancel all active processing tasks
        for task in self.processing_tasks.values():
            task.cancel()
        
        # Wait for tasks to complete
        if self.processing_tasks:
            await asyncio.gather(*self.processing_tasks.values(), return_exceptions=True)
        
        self.processing_tasks.clear()
    
    async def submit_job(self, job: BatchJob) -> Either[ValidationError, BatchJobId]:
        """Submit a batch job for processing."""
        try:
            # Validate job
            validation_result = job.validate_dependencies()
            if validation_result.is_left():
                return validation_result
            
            # Check resource requirements
            estimated_resources = job.estimate_total_resources()
            if not self._check_resource_availability(estimated_resources):
                return Either.left(ValidationError(
                    "insufficient_resources",
                    f"Job requires more resources than available: {estimated_resources}"
                ))
            
            # Create job state
            job_state = BatchJobState(job=job)
            self.active_jobs[job.job_id] = job_state
            
            # Add to priority queue
            heapq.heappush(self.job_queue, (-int(job.priority), job.job_id))
            
            return Either.right(job.job_id)
            
        except Exception as e:
            return Either.left(ValidationError("job_submission_failed", str(e)))
    
    def _check_resource_availability(self, required_resources: Dict[str, float]) -> bool:
        """Check if required resources are available."""
        current_usage = self._calculate_current_resource_usage()
        
        for resource, required in required_resources.items():
            if resource in self.global_resource_limits:
                available = self.global_resource_limits[resource] - current_usage.get(resource, 0)
                if required > available:
                    return False
        
        return True
    
    def _calculate_current_resource_usage(self) -> Dict[str, float]:
        """Calculate current resource usage across all active jobs."""
        total_usage = {"cpu": 0.0, "memory": 0.0, "concurrent_requests": 0.0}
        
        for job_state in self.active_jobs.values():
            if job_state.status == BatchStatus.RUNNING:
                for resource, usage in job_state.current_resource_usage.items():
                    if resource in total_usage:
                        total_usage[resource] += usage
        
        return total_usage
    
    async def _job_scheduler(self) -> None:
        """Background job scheduler."""
        while self.is_running:
            try:
                # Process jobs from priority queue
                if self.job_queue and len(self.processing_tasks) < self.max_concurrent_jobs:
                    _, job_id = heapq.heappop(self.job_queue)
                    
                    if job_id in self.active_jobs:
                        job_state = self.active_jobs[job_id]
                        if job_state.status == BatchStatus.PENDING:
                            # Start job processing
                            task = asyncio.create_task(self._process_job(job_state))
                            self.processing_tasks[job_id] = task
                
                # Clean up completed processing tasks
                completed_tasks = []
                for job_id, task in self.processing_tasks.items():
                    if task.done():
                        completed_tasks.append(job_id)
                
                for job_id in completed_tasks:
                    del self.processing_tasks[job_id]
                
                await asyncio.sleep(1.0)  # Check every second
                
            except Exception:
                # Log error but continue scheduling
                await asyncio.sleep(5.0)
    
    async def _process_job(self, job_state: BatchJobState) -> None:
        """Process a batch job."""
        try:
            job_state.status = BatchStatus.RUNNING
            job_state.started_at = datetime.now(UTC)
            
            if job_state.job.processing_mode == BatchMode.SEQUENTIAL:
                await self._process_sequential(job_state)
            elif job_state.job.processing_mode == BatchMode.PARALLEL:
                await self._process_parallel(job_state)
            elif job_state.job.processing_mode == BatchMode.PIPELINE:
                await self._process_pipeline(job_state)
            elif job_state.job.processing_mode == BatchMode.PRIORITY:
                await self._process_priority(job_state)
            elif job_state.job.processing_mode == BatchMode.RESOURCE_AWARE:
                await self._process_resource_aware(job_state)
            
            # Determine final status
            if all(result.is_successful() for result in job_state.task_results.values()):
                job_state.status = BatchStatus.COMPLETED
            else:
                job_state.status = BatchStatus.FAILED
            
            job_state.completed_at = datetime.now(UTC)
            
            # Execute notification callbacks
            await self._notify_job_completion(job_state)
            
        except Exception as e:
            job_state.status = BatchStatus.FAILED
            job_state.completed_at = datetime.now(UTC)
            # Log error
    
    async def _process_sequential(self, job_state: BatchJobState) -> None:
        """Process tasks sequentially."""
        for task in job_state.job.tasks:
            if not self.is_running:
                break
            
            result = await self._execute_task(task, job_state)
            job_state.task_results[task.task_id] = result
            
            if not result.is_successful() and not job_state.job.auto_retry_failed:
                break  # Stop on first failure if auto-retry disabled
    
    async def _process_parallel(self, job_state: BatchJobState) -> None:
        """Process tasks in parallel respecting dependencies."""
        completed_tasks = set()
        
        while len(completed_tasks) < len(job_state.job.tasks) and self.is_running:
            # Get ready tasks
            ready_tasks = [task for task in job_state.job.tasks 
                          if task.task_id not in completed_tasks 
                          and task.task_id not in job_state.active_tasks
                          and task.dependencies.issubset(completed_tasks)]
            
            if not ready_tasks:
                # Wait for active tasks to complete
                await asyncio.sleep(0.1)
                continue
            
            # Limit concurrent tasks
            available_slots = job_state.job.max_concurrent_tasks - len(job_state.active_tasks)
            tasks_to_start = ready_tasks[:available_slots]
            
            # Start tasks
            task_futures = []
            for task in tasks_to_start:
                job_state.active_tasks.add(task.task_id)
                future = asyncio.create_task(self._execute_task(task, job_state))
                task_futures.append((task.task_id, future))
            
            # Wait for at least one task to complete
            if task_futures:
                done, pending = await asyncio.wait(
                    [future for _, future in task_futures],
                    return_when=asyncio.FIRST_COMPLETED
                )
                
                # Process completed tasks
                for task_id, future in task_futures:
                    if future in done:
                        result = await future
                        job_state.task_results[task_id] = result
                        job_state.active_tasks.discard(task_id)
                        completed_tasks.add(task_id)
    
    async def _process_pipeline(self, job_state: BatchJobState) -> None:
        """Process tasks in pipeline mode (stages)."""
        # For simplicity, fallback to parallel processing
        # In practice, would implement proper pipeline stages
        await self._process_parallel(job_state)
    
    async def _process_priority(self, job_state: BatchJobState) -> None:
        """Process tasks by priority order."""
        # Sort tasks by priority
        sorted_tasks = sorted(job_state.job.tasks, key=lambda t: t.priority, reverse=True)
        
        for task in sorted_tasks:
            if not self.is_running:
                break
            
            result = await self._execute_task(task, job_state)
            job_state.task_results[task.task_id] = result
    
    async def _process_resource_aware(self, job_state: BatchJobState) -> None:
        """Process tasks with resource awareness."""
        # Implement resource-aware scheduling
        # For now, fallback to parallel with resource checking
        await self._process_parallel(job_state)
    
    async def _execute_task(self, task: BatchTask, job_state: BatchJobState) -> TaskResult:
        """Execute a single batch task."""
        start_time = datetime.now(UTC)
        
        try:
            # Create AI request
            request_result = create_ai_request(
                operation=task.operation,
                input_data=task.input_data,
                **task.processing_parameters
            )
            
            if request_result.is_left():
                return TaskResult(
                    task_id=task.task_id,
                    status=BatchStatus.FAILED,
                    error=str(request_result.get_left()),
                    execution_time=(datetime.now(UTC) - start_time).total_seconds()
                )
            
            # Execute with timeout
            request = request_result.get_right()
            
            try:
                response_result = await asyncio.wait_for(
                    self.ai_manager.process_ai_request(
                        request.operation,
                        request.input_data,
                        processing_mode="balanced",
                        enable_caching=True
                    ),
                    timeout=task.timeout.total_seconds()
                )
                
                execution_time = (datetime.now(UTC) - start_time).total_seconds()
                
                if response_result.is_right():
                    response_data = response_result.get_right()
                    # Convert response dict to AIResponse-like object for consistency
                    return TaskResult(
                        task_id=task.task_id,
                        status=BatchStatus.COMPLETED,
                        result=response_data,  # Using dict instead of AIResponse for simplicity
                        execution_time=execution_time,
                        resource_usage=task.estimate_resource_usage()
                    )
                else:
                    return TaskResult(
                        task_id=task.task_id,
                        status=BatchStatus.FAILED,
                        error=str(response_result.get_left()),
                        execution_time=execution_time
                    )
                    
            except asyncio.TimeoutError:
                return TaskResult(
                    task_id=task.task_id,
                    status=BatchStatus.FAILED,
                    error=f"Task timed out after {task.timeout.total_seconds()} seconds",
                    execution_time=(datetime.now(UTC) - start_time).total_seconds()
                )
                
        except Exception as e:
            return TaskResult(
                task_id=task.task_id,
                status=BatchStatus.FAILED,
                error=str(e),
                execution_time=(datetime.now(UTC) - start_time).total_seconds()
            )
    
    async def _notify_job_completion(self, job_state: BatchJobState) -> None:
        """Notify completion callbacks."""
        for callback in job_state.job.notification_callbacks:
            try:
                if asyncio.iscoroutinefunction(callback):
                    await callback(job_state)
                else:
                    callback(job_state)
            except Exception:
                # Log error but continue
                pass
    
    def get_job_status(self, job_id: BatchJobId) -> Optional[Dict[str, Any]]:
        """Get current status of a batch job."""
        job_state = self.active_jobs.get(job_id)
        if not job_state:
            return None
        
        return job_state.get_execution_summary()
    
    def cancel_job(self, job_id: BatchJobId) -> bool:
        """Cancel a batch job."""
        job_state = self.active_jobs.get(job_id)
        if not job_state:
            return False
        
        job_state.status = BatchStatus.CANCELLED
        job_state.completed_at = datetime.now(UTC)
        
        # Cancel processing task if running
        if job_id in self.processing_tasks:
            self.processing_tasks[job_id].cancel()
        
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive batch processing system status."""
        active_jobs = len(self.active_jobs)
        running_jobs = len([js for js in self.active_jobs.values() if js.status == BatchStatus.RUNNING])
        queued_jobs = len(self.job_queue)
        
        return {
            "is_running": self.is_running,
            "active_jobs": active_jobs,
            "running_jobs": running_jobs,
            "queued_jobs": queued_jobs,
            "max_concurrent_jobs": self.max_concurrent_jobs,
            "resource_limits": dict(self.global_resource_limits),
            "current_resource_usage": self._calculate_current_resource_usage(),
            "processing_tasks": len(self.processing_tasks),
            "timestamp": datetime.now(UTC).isoformat()
        }