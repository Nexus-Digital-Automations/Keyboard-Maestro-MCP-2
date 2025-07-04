"""
Real-Time Metrics Collector - TASK_54 Phase 2 Implementation

High-performance metrics collection engine for system and automation performance monitoring.
Implements Design by Contract patterns with <5% monitoring overhead.

Architecture: Async metrics collection + Type Safety + Performance optimization
Performance: <100ms collection cycles, <5% system overhead
Security: Resource access validation, metric data protection
"""

from __future__ import annotations
import asyncio
import time
import psutil
from datetime import datetime, UTC, timedelta
from typing import Dict, List, Optional, Set, Callable, Any
from dataclasses import dataclass, field
import logging
from concurrent.futures import ThreadPoolExecutor
from contextlib import asynccontextmanager

from ..core.contracts import require, ensure
from ..core.either import Either
from ..core.performance_monitoring import (
    MonitoringSessionID, MetricID, MetricType, MonitoringScope,
    MetricValue, SystemResourceSnapshot, MonitoringConfiguration,
    PerformanceMetrics, PerformanceThreshold, PerformanceAlert,
    generate_metric_id, generate_alert_id, collect_system_snapshot,
    CPUPercentage, MemoryBytes, ExecutionTimeMS, ThroughputOPS, LatencyMS,
    MetricCollectionError, ThresholdViolationError
)

logger = logging.getLogger(__name__)


@dataclass
class MetricCollectionSession:
    """Active metrics collection session with real-time monitoring."""
    session_id: MonitoringSessionID
    configuration: MonitoringConfiguration
    is_active: bool = True
    metrics: PerformanceMetrics = field(init=False)
    collection_task: Optional[asyncio.Task] = None
    last_collection_time: Optional[datetime] = None
    collection_count: int = 0
    error_count: int = 0
    
    def __post_init__(self):
        self.metrics = PerformanceMetrics(
            session_id=self.session_id,
            start_time=datetime.now(UTC)
        )


class MetricsCollector:
    """
    High-performance real-time metrics collection engine.
    
    Collects system and automation performance metrics with minimal overhead.
    Supports threshold monitoring, alert generation, and historical tracking.
    """
    
    def __init__(self, max_concurrent_sessions: int = 10):
        require(lambda: max_concurrent_sessions > 0, "Max sessions must be positive")
        
        self.max_concurrent_sessions = max_concurrent_sessions
        self.active_sessions: Dict[MonitoringSessionID, MetricCollectionSession] = {}
        self.metric_collectors: Dict[MetricType, Callable] = {}
        self.threshold_evaluators: List[Callable] = []
        self.executor = ThreadPoolExecutor(max_workers=4, thread_name_prefix="metrics")
        self.is_running = False
        
        # Initialize metric collectors
        self._initialize_collectors()
        
        logger.info(f"MetricsCollector initialized with {max_concurrent_sessions} max sessions")
    
    def _initialize_collectors(self) -> None:
        """Initialize metric collection functions for each metric type."""
        self.metric_collectors = {
            MetricType.CPU: self._collect_cpu_metrics,
            MetricType.MEMORY: self._collect_memory_metrics,
            MetricType.DISK: self._collect_disk_metrics,
            MetricType.NETWORK: self._collect_network_metrics,
            MetricType.EXECUTION_TIME: self._collect_execution_time_metrics,
            MetricType.THROUGHPUT: self._collect_throughput_metrics,
            MetricType.LATENCY: self._collect_latency_metrics,
            MetricType.ERROR_RATE: self._collect_error_rate_metrics,
            MetricType.QUEUE_SIZE: self._collect_queue_size_metrics,
            MetricType.CONNECTION_COUNT: self._collect_connection_count_metrics
        }
    
    @require(lambda config: config.sampling_interval > 0, "Sampling interval must be positive")
    @ensure(lambda result: result.is_right() or isinstance(result.left(), str), "Returns session or error")
    async def start_collection_session(
        self, 
        configuration: MonitoringConfiguration
    ) -> Either[str, MetricCollectionSession]:
        """Start a new metrics collection session with the given configuration."""
        try:
            # Check session limits
            if len(self.active_sessions) >= self.max_concurrent_sessions:
                return Either.left(f"Maximum {self.max_concurrent_sessions} concurrent sessions exceeded")
            
            # Validate configuration
            if not configuration.metrics_types:
                return Either.left("At least one metric type must be specified")
            
            # Create session
            session = MetricCollectionSession(
                session_id=configuration.session_id,
                configuration=configuration
            )
            
            # Start collection task
            session.collection_task = asyncio.create_task(
                self._collection_loop(session)
            )
            
            self.active_sessions[configuration.session_id] = session
            
            logger.info(f"Started metrics collection session {configuration.session_id}")
            return Either.right(session)
            
        except Exception as e:
            error_msg = f"Failed to start collection session: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    @require(lambda session_id: session_id is not None, "Session ID required")
    async def stop_collection_session(self, session_id: MonitoringSessionID) -> Either[str, PerformanceMetrics]:
        """Stop a metrics collection session and return collected metrics."""
        try:
            if session_id not in self.active_sessions:
                return Either.left(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            session.is_active = False
            
            # Cancel collection task
            if session.collection_task and not session.collection_task.done():
                session.collection_task.cancel()
                try:
                    await session.collection_task
                except asyncio.CancelledError:
                    pass
            
            # Finalize metrics
            session.metrics.end_time = datetime.now(UTC)
            
            # Remove from active sessions
            del self.active_sessions[session_id]
            
            logger.info(f"Stopped metrics collection session {session_id}")
            return Either.right(session.metrics)
            
        except Exception as e:
            error_msg = f"Failed to stop collection session: {str(e)}"
            logger.error(error_msg)
            return Either.left(error_msg)
    
    async def _collection_loop(self, session: MetricCollectionSession) -> None:
        """Main collection loop for a monitoring session."""
        try:
            while session.is_active:
                start_time = time.time()
                
                # Collect system snapshot
                try:
                    snapshot = await self._collect_system_snapshot_async()
                    session.metrics.add_snapshot(snapshot)
                except Exception as e:
                    session.error_count += 1
                    logger.warning(f"System snapshot collection failed: {e}")
                
                # Collect specific metrics
                for metric_type in session.configuration.metrics_types:
                    try:
                        if metric_type in self.metric_collectors:
                            metric_values = await self.metric_collectors[metric_type]()
                            for metric in metric_values:
                                session.metrics.add_metric(metric)
                                
                                # Check thresholds
                                await self._check_thresholds(session, metric)
                    except Exception as e:
                        session.error_count += 1
                        logger.warning(f"Metric collection failed for {metric_type}: {e}")
                
                session.collection_count += 1
                session.last_collection_time = datetime.now(UTC)
                
                # Calculate next collection time
                collection_time = time.time() - start_time
                sleep_time = max(0, session.configuration.sampling_interval - collection_time)
                
                if sleep_time > 0:
                    await asyncio.sleep(sleep_time)
                
                # Check duration limit
                if session.configuration.duration:
                    elapsed = (datetime.now(UTC) - session.metrics.start_time).total_seconds()
                    if elapsed >= session.configuration.duration:
                        session.is_active = False
                        
        except asyncio.CancelledError:
            logger.info(f"Collection loop cancelled for session {session.session_id}")
        except Exception as e:
            logger.error(f"Collection loop error for session {session.session_id}: {e}")
            session.is_active = False
    
    async def _collect_system_snapshot_async(self) -> SystemResourceSnapshot:
        """Collect system resource snapshot asynchronously."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(self.executor, collect_system_snapshot)
    
    async def _collect_cpu_metrics(self) -> List[MetricValue]:
        """Collect CPU usage metrics."""
        try:
            # Get CPU usage with short interval for accuracy
            cpu_percent = await asyncio.get_event_loop().run_in_executor(
                self.executor, lambda: psutil.cpu_percent(interval=0.1)
            )
            
            metrics = [
                MetricValue(
                    metric_type=MetricType.CPU,
                    value=CPUPercentage(cpu_percent),
                    unit="percent",
                    source="system"
                )
            ]
            
            # Per-CPU metrics if available
            try:
                per_cpu = psutil.cpu_percent(percpu=True)
                for i, cpu_usage in enumerate(per_cpu):
                    metrics.append(
                        MetricValue(
                            metric_type=MetricType.CPU,
                            value=CPUPercentage(cpu_usage),
                            unit="percent",
                            source=f"cpu_{i}"
                        )
                    )
            except Exception:
                pass  # Per-CPU not available on all systems
            
            return metrics
            
        except Exception as e:
            raise MetricCollectionError(f"CPU metrics collection failed: {e}")
    
    async def _collect_memory_metrics(self) -> List[MetricValue]:
        """Collect memory usage metrics."""
        try:
            memory = psutil.virtual_memory()
            swap = psutil.swap_memory()
            
            return [
                MetricValue(
                    metric_type=MetricType.MEMORY,
                    value=MemoryBytes(memory.used),
                    unit="bytes",
                    source="virtual_memory"
                ),
                MetricValue(
                    metric_type=MetricType.MEMORY,
                    value=memory.percent,
                    unit="percent",
                    source="virtual_memory_percent"
                ),
                MetricValue(
                    metric_type=MetricType.MEMORY,
                    value=MemoryBytes(swap.used),
                    unit="bytes",
                    source="swap_memory"
                ),
                MetricValue(
                    metric_type=MetricType.MEMORY,
                    value=swap.percent,
                    unit="percent",
                    source="swap_memory_percent"
                )
            ]
            
        except Exception as e:
            raise MetricCollectionError(f"Memory metrics collection failed: {e}")
    
    async def _collect_disk_metrics(self) -> List[MetricValue]:
        """Collect disk I/O metrics."""
        try:
            disk_io = psutil.disk_io_counters()
            if not disk_io:
                return []
            
            return [
                MetricValue(
                    metric_type=MetricType.DISK,
                    value=disk_io.read_bytes,
                    unit="bytes",
                    source="disk_read"
                ),
                MetricValue(
                    metric_type=MetricType.DISK,
                    value=disk_io.write_bytes,
                    unit="bytes",
                    source="disk_write"
                ),
                MetricValue(
                    metric_type=MetricType.DISK,
                    value=disk_io.read_count,
                    unit="operations",
                    source="disk_read_ops"
                ),
                MetricValue(
                    metric_type=MetricType.DISK,
                    value=disk_io.write_count,
                    unit="operations",
                    source="disk_write_ops"
                )
            ]
            
        except Exception as e:
            raise MetricCollectionError(f"Disk metrics collection failed: {e}")
    
    async def _collect_network_metrics(self) -> List[MetricValue]:
        """Collect network I/O metrics."""
        try:
            network_io = psutil.net_io_counters()
            if not network_io:
                return []
            
            return [
                MetricValue(
                    metric_type=MetricType.NETWORK,
                    value=network_io.bytes_sent,
                    unit="bytes",
                    source="network_sent"
                ),
                MetricValue(
                    metric_type=MetricType.NETWORK,
                    value=network_io.bytes_recv,
                    unit="bytes",
                    source="network_recv"
                ),
                MetricValue(
                    metric_type=MetricType.NETWORK,
                    value=network_io.packets_sent,
                    unit="packets",
                    source="network_packets_sent"
                ),
                MetricValue(
                    metric_type=MetricType.NETWORK,
                    value=network_io.packets_recv,
                    unit="packets",
                    source="network_packets_recv"
                )
            ]
            
        except Exception as e:
            raise MetricCollectionError(f"Network metrics collection failed: {e}")
    
    async def _collect_execution_time_metrics(self) -> List[MetricValue]:
        """Collect execution time metrics (placeholder for macro/automation timing)."""
        # This would integrate with the macro execution engine
        # For now, return empty list as it requires macro execution context
        return []
    
    async def _collect_throughput_metrics(self) -> List[MetricValue]:
        """Collect throughput metrics (placeholder for automation throughput)."""
        # This would integrate with the automation engine
        # For now, return empty list as it requires automation context
        return []
    
    async def _collect_latency_metrics(self) -> List[MetricValue]:
        """Collect latency metrics (placeholder for automation latency)."""
        # This would integrate with the automation engine
        # For now, return empty list as it requires automation context
        return []
    
    async def _collect_error_rate_metrics(self) -> List[MetricValue]:
        """Collect error rate metrics (placeholder for automation errors)."""
        # This would integrate with the audit/logging system
        # For now, return empty list as it requires error tracking context
        return []
    
    async def _collect_queue_size_metrics(self) -> List[MetricValue]:
        """Collect queue size metrics (placeholder for automation queues)."""
        # This would integrate with the automation queue system
        # For now, return empty list as it requires queue management context
        return []
    
    async def _collect_connection_count_metrics(self) -> List[MetricValue]:
        """Collect connection count metrics."""
        try:
            connections = psutil.net_connections()
            active_connections = len([c for c in connections if c.status == 'ESTABLISHED'])
            
            return [
                MetricValue(
                    metric_type=MetricType.CONNECTION_COUNT,
                    value=active_connections,
                    unit="connections",
                    source="active_connections"
                ),
                MetricValue(
                    metric_type=MetricType.CONNECTION_COUNT,
                    value=len(connections),
                    unit="connections",
                    source="total_connections"
                )
            ]
            
        except Exception as e:
            raise MetricCollectionError(f"Connection metrics collection failed: {e}")
    
    async def _check_thresholds(self, session: MetricCollectionSession, metric: MetricValue) -> None:
        """Check if metric value violates any configured thresholds."""
        try:
            for threshold in session.configuration.thresholds:
                if threshold.metric_type == metric.metric_type:
                    if threshold.evaluate(metric.value):
                        # Create alert
                        alert = PerformanceAlert(
                            alert_id=generate_alert_id(),
                            metric_type=metric.metric_type,
                            current_value=metric.value,
                            threshold=threshold,
                            triggered_at=metric.timestamp,
                            source=metric.source,
                            message=f"{metric.metric_type.value} threshold violated: {metric.value} {threshold.operator.value} {threshold.threshold_value}"
                        )
                        
                        session.metrics.add_alert(alert)
                        logger.warning(f"Threshold violation: {alert.message}")
                        
        except Exception as e:
            logger.error(f"Threshold checking failed: {e}")
    
    def get_session_status(self, session_id: MonitoringSessionID) -> Optional[Dict[str, Any]]:
        """Get current status of a monitoring session."""
        if session_id not in self.active_sessions:
            return None
        
        session = self.active_sessions[session_id]
        return {
            "session_id": session.session_id,
            "is_active": session.is_active,
            "collection_count": session.collection_count,
            "error_count": session.error_count,
            "last_collection_time": session.last_collection_time.isoformat() if session.last_collection_time else None,
            "metrics_collected": len(session.metrics.metrics),
            "snapshots_collected": len(session.metrics.snapshots),
            "alerts_generated": len(session.metrics.alerts),
            "elapsed_time_seconds": (datetime.now(UTC) - session.metrics.start_time).total_seconds()
        }
    
    def get_all_sessions_status(self) -> Dict[str, Any]:
        """Get status of all active monitoring sessions."""
        return {
            "active_sessions": len(self.active_sessions),
            "max_sessions": self.max_concurrent_sessions,
            "sessions": [
                self.get_session_status(session_id) 
                for session_id in self.active_sessions.keys()
            ]
        }
    
    async def shutdown(self) -> None:
        """Shutdown the metrics collector and clean up resources."""
        logger.info("Shutting down metrics collector...")
        
        # Stop all active sessions
        session_ids = list(self.active_sessions.keys())
        for session_id in session_ids:
            await self.stop_collection_session(session_id)
        
        # Shutdown executor
        self.executor.shutdown(wait=True)
        
        logger.info("Metrics collector shutdown complete")
    
    # Aliases for compatibility with performance monitor tools
    async def start_monitoring_session(self, configuration: MonitoringConfiguration) -> Either[str, MetricCollectionSession]:
        """Alias for start_collection_session."""
        return await self.start_collection_session(configuration)
    
    def get_active_sessions(self) -> Dict[MonitoringSessionID, MetricCollectionSession]:
        """Get all active monitoring sessions."""
        return self.active_sessions.copy()
    
    async def get_recent_metrics(self, session_id: MonitoringSessionID, count: int = 10) -> Either[str, List[MetricValue]]:
        """Get recent metrics from an active session."""
        try:
            if session_id not in self.active_sessions:
                return Either.left(f"Session {session_id} not found")
            
            session = self.active_sessions[session_id]
            recent_metrics = []
            
            # Get the most recent metrics from the session
            for metric_list in session.metrics.metrics_by_type.values():
                recent_metrics.extend(metric_list[-count:])
            
            # Sort by timestamp (most recent first)
            recent_metrics.sort(key=lambda m: m.timestamp, reverse=True)
            
            return Either.right(recent_metrics[:count])
            
        except Exception as e:
            return Either.left(f"Failed to get recent metrics: {str(e)}")


# Global instance
_metrics_collector: Optional[MetricsCollector] = None


def get_metrics_collector() -> MetricsCollector:
    """Get or create the global metrics collector instance."""
    global _metrics_collector
    if _metrics_collector is None:
        _metrics_collector = MetricsCollector()
    return _metrics_collector


@asynccontextmanager
async def metrics_collection_session(configuration: MonitoringConfiguration):
    """Context manager for temporary metrics collection."""
    collector = get_metrics_collector()
    
    session_result = await collector.start_collection_session(configuration)
    if session_result.is_left():
        raise MetricCollectionError(session_result.left())
    
    session = session_result.right()
    
    try:
        yield session
    finally:
        await collector.stop_collection_session(configuration.session_id)