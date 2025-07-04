"""
Edge Computing Processor Tests - TASK_65 Phase 4 Testing

Comprehensive test suite for IoT edge computing processor with distributed processing,
task scheduling, cluster management, and performance optimization validation.

Architecture: Edge Computing + Local Processing + Distributed Analytics + Real-Time Processing
Performance: <50ms local processing, <100ms edge analytics, <200ms distributed coordination
Security: Edge encryption, secure processing, local data protection, edge authentication
"""

import pytest
import asyncio
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Any, Optional
from unittest.mock import Mock, AsyncMock, patch, MagicMock

from src.iot.edge_processor import (
    EdgeProcessor,
    EdgeComputeTask,
    EdgeCluster,
    ProcessingResult,
    EdgeProcessingMode,
    EdgeTaskPriority,
    create_edge_task,
    create_edge_cluster
)
from src.core.either import Either
from src.core.errors import ValidationError, SecurityError
from src.core.iot_architecture import DeviceId, IoTIntegrationError


class TestEdgeComputeTask:
    """Test EdgeComputeTask data structure and methods."""
    
    def test_edge_task_creation(self):
        """Test edge compute task creation with valid parameters."""
        task = EdgeComputeTask(
            task_id="test_task_001",
            task_name="Test Processing Task",
            device_id="device_123",
            processing_mode=EdgeProcessingMode.LOCAL,
            priority=EdgeTaskPriority.HIGH,
            data_size=1024,
            estimated_compute_time=0.5,
            required_memory=512
        )
        
        assert task.task_id == "test_task_001"
        assert task.task_name == "Test Processing Task"
        assert task.device_id == "device_123"
        assert task.processing_mode == EdgeProcessingMode.LOCAL
        assert task.priority == EdgeTaskPriority.HIGH
        assert task.data_size == 1024
        assert task.estimated_compute_time == 0.5
        assert task.required_memory == 512
        assert task.retry_count == 0
        assert task.max_retries == 3
    
    def test_task_expiration_check(self):
        """Test task expiration validation."""
        # Task without deadline should not be expired
        task = EdgeComputeTask(
            task_id="test_task_002",
            task_name="Non-expiring Task",
            device_id="device_456",
            processing_mode=EdgeProcessingMode.LOCAL,
            priority=EdgeTaskPriority.NORMAL,
            data_size=512,
            estimated_compute_time=0.1,
            required_memory=256
        )
        
        assert not task.is_expired()
        
        # Task with future deadline should not be expired
        task.deadline = datetime.now(UTC) + timedelta(minutes=5)
        assert not task.is_expired()
        
        # Task with past deadline should be expired
        task.deadline = datetime.now(UTC) - timedelta(minutes=1)
        assert task.is_expired()
    
    def test_task_retry_logic(self):
        """Test task retry capability."""
        task = EdgeComputeTask(
            task_id="test_task_003",
            task_name="Retry Test Task",
            device_id="device_789",
            processing_mode=EdgeProcessingMode.DISTRIBUTED,
            priority=EdgeTaskPriority.CRITICAL,
            data_size=2048,
            estimated_compute_time=1.0,
            required_memory=1024,
            max_retries=2
        )
        
        # Should be able to retry initially
        assert task.can_retry()
        
        # After one retry, should still be able to retry
        task.retry_count = 1
        assert task.can_retry()
        
        # After max retries, should not be able to retry
        task.retry_count = 2
        assert not task.can_retry()
        
        # Beyond max retries, should not be able to retry
        task.retry_count = 3
        assert not task.can_retry()


class TestEdgeCluster:
    """Test EdgeCluster data structure and methods."""
    
    def test_edge_cluster_creation(self):
        """Test edge cluster creation with valid configuration."""
        cluster = EdgeCluster(
            cluster_id="cluster_001",
            nodes=["node_001", "node_002", "node_003"],
            total_compute_capacity=10.0,
            available_capacity=8.0,
            total_memory=16384,
            available_memory=12288,
            network_latency=0.05,
            cluster_health=0.95
        )
        
        assert cluster.cluster_id == "cluster_001"
        assert len(cluster.nodes) == 3
        assert cluster.total_compute_capacity == 10.0
        assert cluster.available_capacity == 8.0
        assert cluster.total_memory == 16384
        assert cluster.available_memory == 12288
        assert cluster.network_latency == 0.05
        assert cluster.cluster_health == 0.95
        assert cluster.load_balancing_enabled is True
        assert cluster.auto_scaling_enabled is True
    
    def test_cluster_capacity_check(self):
        """Test cluster capacity validation for tasks."""
        cluster = EdgeCluster(
            cluster_id="capacity_test",
            nodes=["node_001"],
            total_compute_capacity=5.0,
            available_capacity=3.0,
            total_memory=8192,
            available_memory=4096,
            network_latency=0.1,
            cluster_health=0.8
        )
        
        # Task within capacity should be accepted
        small_task = EdgeComputeTask(
            task_id="small_task",
            task_name="Small Task",
            device_id="device_001",
            processing_mode=EdgeProcessingMode.LOCAL,
            priority=EdgeTaskPriority.NORMAL,
            data_size=512,
            estimated_compute_time=2.0,
            required_memory=2048
        )
        
        assert cluster.has_capacity(small_task)
        
        # Task exceeding capacity should be rejected
        large_task = EdgeComputeTask(
            task_id="large_task",
            task_name="Large Task",
            device_id="device_002",
            processing_mode=EdgeProcessingMode.DISTRIBUTED,
            priority=EdgeTaskPriority.HIGH,
            data_size=4096,
            estimated_compute_time=5.0,
            required_memory=8192
        )
        
        assert not cluster.has_capacity(large_task)


class TestEdgeProcessor:
    """Test EdgeProcessor main functionality."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = EdgeProcessor()
        self.test_cluster = create_edge_cluster("test_cluster", node_count=3)
        self.test_task = create_edge_task(
            device_id="test_device",
            task_name="Test Task",
            processing_mode=EdgeProcessingMode.LOCAL,
            priority=EdgeTaskPriority.NORMAL
        )
    
    @pytest.mark.asyncio
    async def test_register_edge_cluster_success(self):
        """Test successful edge cluster registration."""
        result = await self.processor.register_edge_cluster(self.test_cluster)
        
        assert result.is_success()
        result_data = result.value
        
        assert result_data["success"] is True
        assert result_data["cluster_info"]["cluster_id"] == "test_cluster"
        assert result_data["cluster_info"]["nodes"] == 3
        assert result_data["total_clusters"] == 1
        assert "test_cluster" in self.processor.edge_clusters
    
    @pytest.mark.asyncio
    async def test_register_cluster_invalid_capacity(self):
        """Test cluster registration with invalid capacity."""
        invalid_cluster = EdgeCluster(
            cluster_id="invalid_cluster",
            nodes=["node_001"],
            total_compute_capacity=0.0,  # Invalid capacity
            available_capacity=0.0,
            total_memory=1024,
            available_memory=512,
            network_latency=0.1,
            cluster_health=0.8
        )
        
        result = await self.processor.register_edge_cluster(invalid_cluster)
        
        assert result.is_error()
        assert "compute capacity must be positive" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_register_cluster_low_health(self):
        """Test cluster registration with low health score."""
        unhealthy_cluster = EdgeCluster(
            cluster_id="unhealthy_cluster",
            nodes=["node_001"],
            total_compute_capacity=5.0,
            available_capacity=3.0,
            total_memory=4096,
            available_memory=2048,
            network_latency=0.1,
            cluster_health=0.5  # Below minimum 70%
        )
        
        result = await self.processor.register_edge_cluster(unhealthy_cluster)
        
        assert result.is_error()
        assert "health too low" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_submit_processing_task_success(self):
        """Test successful task submission."""
        # Register cluster first
        await self.processor.register_edge_cluster(self.test_cluster)
        
        result = await self.processor.submit_processing_task(self.test_task)
        
        assert result.is_success()
        result_data = result.value
        
        assert result_data["success"] is True
        assert result_data["submission_info"]["task_id"] == self.test_task.task_id
        assert result_data["submission_info"]["task_name"] == self.test_task.task_name
        assert result_data["submission_info"]["assigned_cluster"] == "test_cluster"
        assert self.test_task.task_id in self.processor.active_tasks
        assert len(self.processor.task_queue) == 1
    
    @pytest.mark.asyncio
    async def test_submit_expired_task(self):
        """Test submission of expired task."""
        # Create expired task
        expired_task = create_edge_task(
            device_id="expired_device",
            task_name="Expired Task"
        )
        expired_task.deadline = datetime.now(UTC) - timedelta(minutes=1)
        
        result = await self.processor.submit_processing_task(expired_task)
        
        assert result.is_error()
        assert "expired" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_submit_oversized_task(self):
        """Test submission of task with excessive data size."""
        oversized_task = create_edge_task(
            device_id="oversized_device",
            task_name="Oversized Task",
            data_size=200_000_000  # 200MB - exceeds 100MB limit
        )
        
        result = await self.processor.submit_processing_task(oversized_task)
        
        assert result.is_error()
        assert "data size too large" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_submit_task_no_suitable_cluster(self):
        """Test task submission when no suitable cluster exists."""
        # Don't register any clusters
        result = await self.processor.submit_processing_task(self.test_task)
        
        assert result.is_error()
        assert "No suitable edge cluster available" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_task_queue_processing(self):
        """Test task queue processing with multiple tasks."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Submit multiple tasks
        tasks = [
            create_edge_task(f"device_{i}", f"Task {i}", priority=EdgeTaskPriority.HIGH if i % 2 == 0 else EdgeTaskPriority.NORMAL)
            for i in range(3)
        ]
        
        for task in tasks:
            await self.processor.submit_processing_task(task)
        
        # Process queue
        results = []
        async for result in self.processor.process_task_queue():
            if result.is_success():
                results.append(result.value)
            if len(results) >= 3:  # Stop after processing all tasks
                break
        
        assert len(results) == 3
        assert all(result.success for result in results)
        assert self.processor.total_tasks_processed == 3
    
    @pytest.mark.asyncio
    async def test_priority_based_scheduling(self):
        """Test that tasks are processed based on priority."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Submit tasks with different priorities
        low_priority_task = create_edge_task("device_low", "Low Priority", priority=EdgeTaskPriority.LOW)
        high_priority_task = create_edge_task("device_high", "High Priority", priority=EdgeTaskPriority.HIGH)
        critical_task = create_edge_task("device_critical", "Critical Task", priority=EdgeTaskPriority.CRITICAL)
        
        # Submit in reverse priority order
        await self.processor.submit_processing_task(low_priority_task)
        await self.processor.submit_processing_task(high_priority_task)
        await self.processor.submit_processing_task(critical_task)
        
        # Verify queue ordering - critical should be first
        assert self.processor.task_queue[0].priority == EdgeTaskPriority.CRITICAL
        assert self.processor.task_queue[1].priority == EdgeTaskPriority.HIGH
        assert self.processor.task_queue[2].priority == EdgeTaskPriority.LOW
    
    @pytest.mark.asyncio
    async def test_task_retry_mechanism(self):
        """Test task retry mechanism when no cluster is available."""
        # Submit task without registering cluster
        result = await self.processor.submit_processing_task(self.test_task)
        assert result.is_error()
        
        # Now register cluster and try processing
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Add task manually to queue for retry testing
        self.processor.task_queue.append(self.test_task)
        self.processor.active_tasks[self.test_task.task_id] = self.test_task
        
        # Process should succeed
        results = []
        async for result in self.processor.process_task_queue():
            if result.is_success():
                results.append(result.value)
                break
        
        assert len(results) == 1
        assert results[0].success
    
    @pytest.mark.asyncio
    async def test_different_processing_modes(self):
        """Test different edge processing modes."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Test each processing mode
        modes = [
            EdgeProcessingMode.LOCAL,
            EdgeProcessingMode.DISTRIBUTED,
            EdgeProcessingMode.HYBRID,
            EdgeProcessingMode.CLOUD_FALLBACK
        ]
        
        for mode in modes:
            task = create_edge_task(
                f"device_{mode.value}",
                f"Task {mode.value}",
                processing_mode=mode
            )
            
            result = await self.processor.submit_processing_task(task)
            assert result.is_success()
        
        # Process all tasks
        results = []
        async for result in self.processor.process_task_queue():
            if result.is_success():
                results.append(result.value)
            if len(results) >= len(modes):
                break
        
        assert len(results) == len(modes)
        assert all(result.success for result in results)
    
    @pytest.mark.asyncio
    async def test_performance_metrics_tracking(self):
        """Test performance metrics tracking."""
        # Register cluster and submit task
        await self.processor.register_edge_cluster(self.test_cluster)
        await self.processor.submit_processing_task(self.test_task)
        
        # Process task
        async for result in self.processor.process_task_queue():
            if result.is_success():
                break
        
        # Check metrics
        assert self.processor.total_tasks_processed == 1
        assert self.processor.average_processing_time > 0
        assert self.processor.peak_memory_usage > 0
        assert self.processor.cluster_utilization >= 0
    
    @pytest.mark.asyncio
    async def test_get_processing_status(self):
        """Test processing status retrieval."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Submit task
        await self.processor.submit_processing_task(self.test_task)
        
        status = await self.processor.get_processing_status()
        
        assert status["total_clusters"] == 1
        assert status["active_tasks"] == 1
        assert status["queued_tasks"] == 1
        assert status["total_processed"] == 0
        assert "cluster_health" in status
        assert "test_cluster" in status["cluster_health"]
    
    @pytest.mark.asyncio
    async def test_processing_result_structure(self):
        """Test processing result data structure."""
        # Register cluster and submit task
        await self.processor.register_edge_cluster(self.test_cluster)
        await self.processor.submit_processing_task(self.test_task)
        
        # Process task
        async for result in self.processor.process_task_queue():
            if result.is_success():
                processing_result = result.value
                
                # Verify result structure
                assert processing_result.task_id == self.test_task.task_id
                assert processing_result.success is True
                assert processing_result.result_data is not None
                assert processing_result.processing_time > 0
                assert processing_result.memory_used > 0
                assert processing_result.edge_node is not None
                assert processing_result.performance_metrics is not None
                
                # Verify analytics results
                assert "analytics_results" in processing_result.result_data
                assert "edge_insights" in processing_result.result_data
                assert "patterns_detected" in processing_result.result_data["analytics_results"]
                assert "local_processing_efficiency" in processing_result.result_data["edge_insights"]
                
                break


class TestEdgeProcessorHelpers:
    """Test helper functions for edge processing."""
    
    def test_create_edge_task_helper(self):
        """Test create_edge_task helper function."""
        task = create_edge_task(
            device_id="helper_device",
            task_name="Helper Task",
            processing_mode=EdgeProcessingMode.HYBRID,
            priority=EdgeTaskPriority.HIGH,
            data_size=2048,
            compute_time=1.5,
            memory_requirement=4096
        )
        
        assert task.device_id == "helper_device"
        assert task.task_name == "Helper Task"
        assert task.processing_mode == EdgeProcessingMode.HYBRID
        assert task.priority == EdgeTaskPriority.HIGH
        assert task.data_size == 2048
        assert task.estimated_compute_time == 1.5
        assert task.required_memory == 4096
        assert task.task_id.startswith("task_")
    
    def test_create_edge_cluster_helper(self):
        """Test create_edge_cluster helper function."""
        cluster = create_edge_cluster(
            cluster_id="helper_cluster",
            node_count=5,
            compute_capacity=20.0,
            memory_capacity=32768,
            health=0.98
        )
        
        assert cluster.cluster_id == "helper_cluster"
        assert len(cluster.nodes) == 5
        assert cluster.total_compute_capacity == 20.0
        assert cluster.total_memory == 32768
        assert cluster.cluster_health == 0.98
        assert cluster.available_capacity == 16.0  # 80% of total
        assert cluster.available_memory == 26214  # 80% of total
        assert all(node.startswith("node_helper_cluster_") for node in cluster.nodes)


class TestEdgeProcessorConcurrency:
    """Test edge processor concurrency and thread safety."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = EdgeProcessor()
        self.test_cluster = create_edge_cluster("concurrent_cluster", node_count=4)
    
    @pytest.mark.asyncio
    async def test_concurrent_task_submission(self):
        """Test concurrent task submission."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Submit multiple tasks concurrently
        tasks = [
            create_edge_task(f"concurrent_device_{i}", f"Concurrent Task {i}")
            for i in range(10)
        ]
        
        submission_results = await asyncio.gather(*[
            self.processor.submit_processing_task(task)
            for task in tasks
        ])
        
        # All submissions should succeed
        assert all(result.is_success() for result in submission_results)
        assert len(self.processor.active_tasks) == 10
        assert len(self.processor.task_queue) == 10
    
    @pytest.mark.asyncio
    async def test_concurrent_cluster_registration(self):
        """Test concurrent cluster registration."""
        clusters = [
            create_edge_cluster(f"concurrent_cluster_{i}", node_count=2)
            for i in range(5)
        ]
        
        registration_results = await asyncio.gather(*[
            self.processor.register_edge_cluster(cluster)
            for cluster in clusters
        ])
        
        # All registrations should succeed
        assert all(result.is_success() for result in registration_results)
        assert len(self.processor.edge_clusters) == 5
    
    @pytest.mark.asyncio
    async def test_concurrent_processing_and_submission(self):
        """Test concurrent processing and task submission."""
        # Register cluster
        await self.processor.register_edge_cluster(self.test_cluster)
        
        # Submit initial tasks
        initial_tasks = [
            create_edge_task(f"initial_device_{i}", f"Initial Task {i}")
            for i in range(3)
        ]
        
        for task in initial_tasks:
            await self.processor.submit_processing_task(task)
        
        # Start processing and submit more tasks concurrently
        async def process_tasks():
            results = []
            async for result in self.processor.process_task_queue():
                if result.is_success():
                    results.append(result.value)
                if len(results) >= 5:  # Process 5 tasks
                    break
            return results
        
        async def submit_more_tasks():
            additional_tasks = [
                create_edge_task(f"additional_device_{i}", f"Additional Task {i}")
                for i in range(2)
            ]
            
            results = []
            for task in additional_tasks:
                result = await self.processor.submit_processing_task(task)
                results.append(result)
            return results
        
        # Run both operations concurrently
        processing_results, submission_results = await asyncio.gather(
            process_tasks(),
            submit_more_tasks()
        )
        
        assert len(processing_results) == 5
        assert len(submission_results) == 2
        assert all(result.success for result in processing_results)
        assert all(result.is_success() for result in submission_results)


class TestEdgeProcessorErrorHandling:
    """Test comprehensive error handling in edge processor."""
    
    def setup_method(self):
        """Setup test fixtures."""
        self.processor = EdgeProcessor()
    
    @pytest.mark.asyncio
    async def test_exception_handling_in_processing(self):
        """Test exception handling during task processing."""
        # Register cluster
        cluster = create_edge_cluster("error_cluster")
        await self.processor.register_edge_cluster(cluster)
        
        # Create task that will cause processing error
        error_task = create_edge_task("error_device", "Error Task")
        
        # Mock processing method to raise exception
        with patch.object(self.processor, '_execute_task', side_effect=Exception("Processing error")):
            await self.processor.submit_processing_task(error_task)
            
            # Process should handle exception gracefully
            results = []
            async for result in self.processor.process_task_queue():
                results.append(result)
                break
            
            assert len(results) == 1
            assert results[0].is_error()
            assert "Processing error" in str(results[0].error_value)
    
    @pytest.mark.asyncio
    async def test_invalid_cluster_parameters(self):
        """Test handling of invalid cluster parameters."""
        # Test negative memory
        invalid_cluster = EdgeCluster(
            cluster_id="invalid_memory",
            nodes=["node_001"],
            total_compute_capacity=5.0,
            available_capacity=3.0,
            total_memory=-1024,  # Invalid negative memory
            available_memory=512,
            network_latency=0.1,
            cluster_health=0.8
        )
        
        result = await self.processor.register_edge_cluster(invalid_cluster)
        assert result.is_error()
        assert "memory must be positive" in str(result.error_value)
    
    @pytest.mark.asyncio
    async def test_edge_processor_resource_limits(self):
        """Test edge processor resource limit handling."""
        # Register small cluster
        small_cluster = EdgeCluster(
            cluster_id="small_cluster",
            nodes=["node_001"],
            total_compute_capacity=1.0,
            available_capacity=0.5,
            total_memory=512,
            available_memory=256,
            network_latency=0.1,
            cluster_health=0.9
        )
        
        await self.processor.register_edge_cluster(small_cluster)
        
        # Submit task that exceeds cluster capacity
        large_task = create_edge_task(
            "large_device",
            "Large Task",
            compute_time=2.0,  # Exceeds available capacity
            memory_requirement=1024  # Exceeds available memory
        )
        
        result = await self.processor.submit_processing_task(large_task)
        assert result.is_error()


if __name__ == "__main__":
    pytest.main([__file__, "-v"])